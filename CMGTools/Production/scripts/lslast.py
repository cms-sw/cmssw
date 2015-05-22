#!/bin/env python
'''
This script returns the last file matching a pattern in a given directory
'''
import glob
import sys
import os 
from stat import *

allFiles = []
if( len(sys.argv)<2 ):
    allFiles = glob.glob('*')
else:
    allFiles = sys.argv[1:]

# accessing time of last modification
filesWithTime = [(os.stat(file)[ST_MTIME], file) for file in allFiles]
time, lastFile = max( filesWithTime )
print lastFile
