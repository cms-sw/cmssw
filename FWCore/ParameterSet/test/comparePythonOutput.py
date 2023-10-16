#!/usr/bin/env python3
#  This compares the content of two Python files,
#  ignoring trivial differences like order within
#  dictionaries or escape characters

from __future__ import print_function
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("file1",type=str)
parser.add_argument("file2",type=str)
options = parser.parse_args()

cfg1 = eval(file(options.file1).read())
cfg2 = eval(file(options.file2).read())
if cfg1 != cfg2:
    print(options.file1, " and ", options.file2, " do not match")
    k1 = set(cfg1.keys())
    k2 = set(cfg2.keys()) 
    if k1-k2 :
      print("Different keys " , k1-k2) 
    else: 
      print("Keys match ")
      for key in k1:
        # skip schedule, because it could get parentheses
        if cfg1[key] != cfg2[key]: 
          # skip schedule, because it could get parentheses
          if key == "schedule":
            exit(0)
          else:
            print("The value of key ", key , " does not match")

    sys.exit(-1)
print("matched")    
