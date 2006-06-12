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
    print "no match"
    exit(-1)
print "matched"    
