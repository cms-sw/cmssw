import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.parseConfig as cmsParse
from sys import argv

print "import FWCore.ParameterSet.Config as cms"

fileInPath = argv[1]

if fileInPath.endswith('cfg'):
    #print cmsParse.parseCfgFile(fileInPath).dumpPython()
    print cmsParse.dumpCfg(fileInPath)
else:
    #print cmsParse.parseCffFile(fileInPath).dumpPython()
    print  cmsParse.dump(fileInPath)

