#!/usr/bin/env python

#https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide
import numpy as np
import numexpr as ne
a = np.arange(10)
b = np.arange(0, 20, 2)
c = ne.evaluate("2*a+3*b")
print c

