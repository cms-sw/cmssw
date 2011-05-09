#!/usr/bin/env python
# Author: Tae.Jeong.Kim@cern.ch
import os
import re
import sys
import time
import commands

from rpcPopcon_cfg import *

def rfdir(dirname):
	dircmd='ls'
	cmd = '%s %s' % (dircmd, dirname)
	output = commands.getoutput( cmd ).rsplit('\n')
	returnVal = []
	for line in output:
		returnVal.append(line.split()[-1])
	return returnVal


currdir = commands.getoutput('pwd') 
dir = currdir+"/dqmdata"

for fname in rfdir(dir):
  if re.search(".root",fname)!=None:
    print fname
    tempnumber = re.split('_', fname)
    runnumber = re.search('(?<=R000)\w+',tempnumber[2])
    print runnumber.group(0)
    process.readMeFromFile.InputFile = cms.untracked.string(dir+"/"+fname)
    process.source.firstRun = cms.untracked.uint32(int( runnumber.group(0) ))
    process.rpcpopcon.Source.IOVRun = cms.untracked.uint32(int( runnumber.group(0) ))
    out = open(dir+'/rpcPopcon_'+runnumber.group(0)+'_cfg.py','w')
    out.write(process.dumpPython())
    out.close()
    os.system("cmsRun "+dir+"/rpcPopcon_"+runnumber.group(0)+"_cfg.py")
