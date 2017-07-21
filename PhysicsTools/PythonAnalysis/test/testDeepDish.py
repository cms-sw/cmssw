#!/usr/bin/env python
#adapted from http://deepdish.readthedocs.io/en/latest/io.html#dictionaries

import numpy as np
import pandas as pd
import deepdish as dd

df = pd.DataFrame({'int': np.arange(3), 'name': ['zero', 'one', 'two']})

dd.io.save('test.h5', df)

d = dd.io.load('test.h5')

print d
