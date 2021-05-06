#!/usr/bin/env python3

#https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide
from __future__ import print_function
import numpy as np
import numexpr as ne
a = np.arange(10)
b = np.arange(0, 20, 2)
c = ne.evaluate("2*a+3*b")
print(c)

