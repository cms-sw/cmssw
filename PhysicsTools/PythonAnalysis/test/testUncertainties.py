#!/usr/bin/env python

#example taken from https://pypi.python.org/pypi/uncertainties/3.0.1
from uncertainties import ufloat
x = ufloat(2, 0.25)
print x
square = x**2  # Transparent calculations
print square
print square.nominal_value
print square.std_dev  # Standard deviation
print square - x*x
from uncertainties.umath import *  # sin(), etc.
print sin(1+x**2)
print (2*x+1000).derivatives[x]  # Automatic calculation of derivatives
from uncertainties import unumpy  # Array manipulation
random_vars = unumpy.uarray([1, 2], [0.1, 0.2])
print random_vars
print random_vars.mean()
print unumpy.cos(random_vars)
