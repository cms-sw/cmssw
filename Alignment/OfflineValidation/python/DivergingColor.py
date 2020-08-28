import os
import math
import numpy as np

# reference: https://www.kennethmoreland.com/color-maps/ColorMapsExpanded.pdf

# RGB to XYZ matrix
# 0.4124 | 0.2126 | 0.0193
# 0.3576 | 0.7152 | 0.1192
# 0.1805 | 0.0722 | 0.9505

# Inverse
#  3.24063  | -0.968931  |  0.0557101
# -1.53721  |  1.87576   | -0.204021
# -0.498629 |  0.0415175 |  1.057




def fRGB(x):
    if x > 0.00313080495356037152: val = math.pow(x, 1./2.4)*1.055 - 0.055
    else: val = x*12.92
    return val*255.

def fRGBinv(x):
    if x > 0.04045: val = math.pow((x + 0.055)/1.055, 2.4)
    else: val = (x+0.)/12.92
    return val*100
    
def lRGB2sRGB(r, g, b):
    #def rgb_lin(x):
    #    if x > 0.00313080495356037152: val = math.pow(x, 1./2.4)*1.055 - 0.055
    #    else: val = x*12.92
    #    return val*255.
    #return rgb_lin((r+0.)/100.), rgb_lin((g+0.)/100.), rgb_lin((b+0.)/100.)
    return fRGB((r+0.)/100.), fRGB((g+0.)/100.), fRGB((b+0.)/100.)

def sRGB2lRGB(r, g, b):
    #def srgb_lrgb(x):
    #    if x > 0.04045: val = math.pow((x + 0.055)/1.055, 2.4)
    #    else: val = (x+0.)/12.92
    #    return val*100
    #return srgb_lrgb((r+0.)/255.), srgb_lrgb((g+0.)/255.), srgb_lrgb((b+0.)/255.)
    return fRGBinv((r+0.)/255.), fRGBinv((g+0.)/255.), fRGBinv((b+0.)/255.)


def rgb2xyz(r, g, b):
    sr, sg, sb = sRGB2lRGB(r, g, b)
    x = 0.4124*sr + 0.3576*sg + 0.1805*sb
    y = 0.2126*sr + 0.7152*sg + 0.0722*sb
    z = 0.0193*sr + 0.1192*sg + 0.9505*sb
    return x, y, z

def xyz2rgb(x, y, z):
    r =    3.24063*x -  1.53721*y -  0.498629*z
    g =  -0.968931*x +  1.87576*y + 0.0415175*z
    b =  0.0557101*x - 0.204021*y +     1.057*z
    #m = max(max(r, g), b)
    #if m > 1.:
    #    r = (r+0.)/(m+0.)
    #    g = (g+0.)/(m+0.)
    #    b = (b+0.)/(m+0.)
    return lRGB2sRGB(r, g, b)



def F(v):
    if v > 0.008856: return math.pow(v, 1./3.)
    else: return 7.787*v + 16./116.

def Finv(v):
    if v > 0.20689270648: return math.pow(v, 3)
    else: return (v - 16./116.)/7.787


def xyz2Lab(x, y, z, refW):
    xn = refW[0]
    yn = refW[1]
    zn = refW[2]
    #xn, yn, zn = [95.047, 100.0, 108.883]
    #def F(v):
    #    if v > 0.008856: return math.pow(v, 1./3.)
    #    else: return 7.787*v + 16./116.
    L = 116*(F((y+0.)/(yn +0.)) - 16./116.)
    a = 500*(F((x+0.)/(xn+0.)) - F((y+0.)/(yn+0.)))
    b = 200*(F((y+0.)/(yn+0.)) - F((z+0.)/(zn+0.))) 
    return L, a, b

def Lab2xyz(L, a, b, refW):
    xn = refW[0]
    yn = refW[1]
    zn = refW[2]
    #xn, yn, zn = [95.047, 100.0, 108.883]
    #def Finv(v):
    #    if v > 0.20689270648: return math.pow(v, 3)
    #    else: return (v - 16./116.)/7.787
    x = Finv((a+0.)/500. + (L + 16.)/116.)*xn
    y = Finv((L + 16.)/116.)*yn
    z = Finv((L + 16.)/116. - (b+0.)/200.)*zn
    return x, y, z

def Lab2Msh(L, a, b):
    M = math.sqrt(math.pow(L,2) + math.pow(a,2) + math.pow(b,2))
    s = math.acos((L+0.)/(M+0.))
    h = math.atan2(b,a)
    return M, s, h

def Msh2Lab(M, s, h):
    L = M*math.cos(s)
    a = M*math.sin(s)*math.cos(h)
    b = M*math.sin(s)*math.sin(h)
    return L, a, b

def rgb2Msh(r, g, b, refW):
    x, y, z = rgb2xyz(r, g, b)
    xr, yr, zr = rgb2xyz(refW[0], refW[1], refW[2])
    L, a, b = xyz2Lab(x, y, z, [xr, yr, zr])
    return Lab2Msh(L, a, b)

def Msh2rgb(M, s, h, refW):
    xr, yr, zr = rgb2xyz(refW[0], refW[1], refW[2])
    L, a, b = Msh2Lab(M, s, h)
    x, y, z = Lab2xyz(L, a, b, [xr, yr, zr])
    return xyz2rgb(x, y, z)

def AdjustHue(Ms, ss, hs, Mu):
    #print('Adjusting Hue')
    if Ms >= Mu: return hs
    h = ss*math.sqrt(math.pow(Mu, 2.) - math.pow(Ms, 2.))/(Ms*math.sin(ss)+0.)
    if hs > -math.pi/3.: return hs + h
    else: return hs - h

def radDiff(a1, a2):
    diff = abs(a1 - a2)
    if diff > math.pi: return abs(diff - 2*math.pi)
    else: return diff 

#('red: ', (117.34353643868656, 1.099939672641069, 0.698178814103516))
#('blue: ', (137.64998152940237, 1.333915268336423, -0.9374394027523394))

def DivergingColor(col1, col2, white, frac): 
    M1, s1, h1 = rgb2Msh(col1[0], col1[1], col1[2], white)
    M2, s2, h2 = rgb2Msh(col2[0], col2[1], col2[2], white)

    #if s1 > 0.05 and s2 > 0.05 and radDiff(h1,h2) > math.pi/3.:
    if s1 > 0.05 and s2 > 0.05 and abs(h1 - h2) > math.pi/3.:
        Mmid = max(max(M1,M2),88.)
        if frac < .5:
            M2 = Mmid
            s2 = 0.
            h2 = 0.
            frac = 2*frac
        else:
            M1 = Mmid
            s1 = 0.
            h1 = 0.
            frac = 2*frac - 1
    if s1 < 0.05 and s2 > 0.05: h1 = AdjustHue(M2, s2, h2, M1)
    elif s2 < 0.05 and s1 > 0.05: h2 = AdjustHue(M1, s1, h1, M2)

    M = (1 - frac)*M1 + frac*M2
    s = (1 - frac)*s1 + frac*s2
    h = (1 - frac)*h1 + frac*h2
    #print('temp', M, s, h, h1, h2, frac)
    return Msh2rgb(M, s, h, white)

 
if __name__ == '__main__':
    Msh = [83.9912098 ,  0.54009147, -0.18776355]
    Msh_np = np.array(Msh) 
    #red = [243.59789395465015, 146.5213165050506, 192.51678151291404]
    red   = [59, 76, 192]
    blue  = [180, 4, 38]
    white = [1, 1, 1]
    frac = 0.75
    print('my val: ', DivergingColor(blue, red, white, frac))
    print(xyz2rgb(95.047, 100.0, 108.883))

