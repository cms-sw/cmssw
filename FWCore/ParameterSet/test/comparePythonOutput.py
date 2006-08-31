#!/usr/bin/env python
#  This compares the content of two Python files,
#  ignoring trivial differences like order within
#  dictionaries or escape characters

from sys import argv
from sys import exit
if len(argv) < 3:
    print "usage: ",argv[0]," <file1> <file2>"
    exit(-1)
cfg1 = eval(file(argv[1]).read())
cfg2 = eval(file(argv[2]).read())
if cfg1 != cfg2:
    print argv[1], " and ", argv[2], " do not match"
    k1 = set(cfg1.keys())
    k2 = set(cfg2.keys()) 
    if k1-k2 :
      print "Different keys " , k1-k2 
    else: 
      print "Keys match "
      for key in k1:
        # skip schedule, because it could get parentheses
        if cfg1[key] != cfg2[key]: 
          # skip schedule, because it could get parentheses
          if key == "schedule":
            exit(0)
          else:
            print "The value of key ", key , " does not match"

    exit(-1)
print "matched"    
