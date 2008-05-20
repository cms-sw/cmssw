#!/bin/env python
#
from sys import argv
def cfgName2py(cfgName):
    return  cfgName.replace("/data/","/python/").replace(".","_") + ".py"

# for shell-scripters who want a value printed
if __name__=="__main__":
   if len(argv) == 2:
       print cfgName2py(argv[1])
