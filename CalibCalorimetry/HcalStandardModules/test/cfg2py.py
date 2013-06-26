import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.parseConfig as cmsParse
from sys import argv

fileInPath = argv[1]

if fileInPath.endswith('cfg'):
    print cmsParse.dumpCfg(fileInPath)
else:
    print cmsParse.dumpCff(fileInPath)

