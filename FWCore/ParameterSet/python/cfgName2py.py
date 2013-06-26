#!/bin/env python
#
from sys import argv
def cfgName2py(cfgName):
    result = cfgName.replace("/data/","/python/").replace(".","_")
    # do we want to automatically move all test configs to python/test?
    # just do cffs and cfis for now
    if result.endswith('.cff') or result.endswith('.cfi'):
        result = result.replace("/test/", "/python/test/")
    return result + '.py'

# for shell-scripters who want a value printed
if __name__=="__main__":
   if len(argv) == 2:
       print cfgName2py(argv[1])
