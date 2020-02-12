import cv2 as cv
from imutils.perspective import four_point_transform
import pytesseract
import os
from PIL import Image
import argparse
import matplotlib as plt
import numpy as np


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv.resize(image, dim, interpolation=inter)
	return resized


image = cv.imread("page.jpg")
ratio = image.shape[0]/500  # 比例,宽除以500，w自行计算
orig = image.copy()  # copy（）不对原图改变
image = resize(orig, height=500)
cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()

# 预处理操作
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 75, 200)
cv.namedWindow('edged', cv.WINDOW_NORMAL)
cv.imshow('edged', edged)
cv.waitKey(0)
cv.destroyAllWindows()


# 轮廓检测
cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[1]  # 找出图像中的轮廓
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]  # 将轮廓按照面积的大小进行排序，并且选出前5个中最大的轮廓，当多个小票时
for c in cnts:
	peri = cv.arcLength(c, True)  # 周长，闭合
	approx = cv.approxPolyDP(c, 0.02 * peri, True)  # 检测出来的轮廓可能是离散的点，故因在此做近似计算，使其形成一个矩形
	# 做精度控制，原始轮廓到近似轮廓的最大的距离，较小时可能为多边形；较大时可能为矩形
	if len(approx) == 4:  # 如果检测出来的是矩形，则break本段if
		screenCnt = approx
		break
cv.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)  # 绘制轮廓，-1表示全部绘制
cv.imshow('Outline', image)
cv.waitKey(0)
cv.destroyAllWindows()


# 变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)  # 透视变换：摆正图像内容
warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
ref = cv.threshold(warped, 127, 255, cv.THRESH_BINARY)[1]
cv.imwrite('test01.jpg', ref)


image = cv.imread("test01.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
preprocess = 'thresh'  # thresh  #做预处理选项
if preprocess == 'blur':
	gray = cv.medianBlur(gray, 3)
if preprocess == 'thresh':
	gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

filename = "{}.png".format(os.getpid())
cv.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))    # 转化成中文加lang='chi_sim'
print(text)
os.remove(filename)

cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.imshow("Image", image)
cv.namedWindow('Output', cv.WINDOW_NORMAL)
cv.imshow("Output", gray)
cv.waitKey(0)