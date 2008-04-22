import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.parseConfig as cmsParse
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

for fileName in files:
    if fileName.endswith('cfg'):
        print fileName
        newName = fileName.split('.')[0]+'_cfg.py'
        # if it's in a data or test subdirectory, make a python directory
        toks = newName.split('/')
        ntoks = len(toks)
        if ntoks > 1 and (toks[ntoks-2] == 'data' or toks[ntoks-2] == 'test'):
            toks[ntoks-2] = 'python'
            newPath = '/'.join(toks[0:3])
            newName = '/'.join(toks)
        if os.path.exists(newName) and not overwrite:
            continue
        if not os.path.exists(newPath):
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

