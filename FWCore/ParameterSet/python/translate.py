import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.parseConfig as cmsParse
from FWCore.ParameterSet import cfgName2py
import glob
import py_compile
from sys import argv
import os
import os.path


files = list()
overwrite = False
if len(argv) != 2:
    print "Please give either a filename, a directory, or 'local' or 'all'"
elif argv[1] == 'local':
    files = glob.glob("*/*/data/*cf[fi]")
elif argv[1] == 'all':
    cmsswSrc = os.path.expandvars("$CMSSW_BASE/src/")
    cmsswReleaseSrc = os.path.expandvars("$CMSSW_RELEASE_BASE/src/")
    cwd = os.getcwd()
    os.chdir(cmsswReleaseSrc)
    files = glob.glob("*/*/data/*cf[fi]")
    os.chdir(cwd)
elif os.path.isdir(argv[1]):
    globList = argv[1].split('/')
    nToks = len(globList) 
    if nToks < 2:
       globList.append('*')
    if nToks < 3:
       globList.append('data')
    if nToks < 4:
       globList.append('*cf[fi]')
    files = glob.glob('/'.join(globList))
else:
    # single file
    files.append(argv[1])
    overwrite = True
if argv[len(argv)-1] == 'no_overwrite':
    overwrite = False


for fileName in files:
    if fileName.endswith('cfg'):
        newName = cfgName2py.cfgName2py(fileName)
        if os.path.exists(newName) and not overwrite:
            continue
        newPath = os.path.dirname(newName)
        if newPath != '' and not os.path.exists(newPath):
            os.makedirs(newPath)
        f = open(newName, 'w')
        try:
            f.write(cmsParse.dumpCfg(fileName))
        except:
            print "ERROR in "+fileName
            f.close()
            os.remove(newName)
        else:
            f.close()
    else:
        print fileName
        includeNode = cmsParse._IncludeNode(fileName)
        try:
            includeNode.createFile(overwrite)
        except:
            # keep going
            print "ERROR in "+fileName
            os.remove(includeNode.pythonFileName())

