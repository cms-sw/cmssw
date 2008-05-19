#!/bin/env python
#
from sys import argv
def cfgName2py(cfgName):
    return  cfgName.replace("/data/","/python/").replace("/test/","/python/").replace(".","_") + ".py"

# for shell-scripters who want a value printed
print cfgName2py(argv[1])
