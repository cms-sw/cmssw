#!/usr/bin/env python

from __future__ import print_function
import os,sys
import getopt
import commands
import time
import ROOT
import urllib
import string
import optparse
import dataCert

#read arguments to the command line
#configure
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-f', '--firstRun'   ,    dest='firstRun'           , help='first run to process (-1 --> automatic)'  , default='-1')
parser.add_option('-l', '--lastRun'    ,    dest='lastRun'            , help='last run to process (-1 --> automatic)'   , default='-1')
parser.add_option('-P', '--publish'    ,    dest='publish'            , help='publish the results'                      , default='True')
parser.add_option('-p', '--pcl'        ,    dest='usePCL'             , help='use PCL output instead of calibTree'      , default='True')
parser.add_option('-m', '--mode'       ,    dest='calMode'            , help='select the statistics type'      , default='AagBunch')
parser.add_option('-s', '--scriptDir'  ,    dest='scriptDir'          , help='select the scriptDirectory'      , default='')
parser.add_option('-a', '--automatic'  ,    dest='automatic'          , help='set if ran automaticaly'         , default='')
(opt, args) = parser.parse_args()

scriptDir = os.getcwd()
firstRun = int(opt.firstRun)
lastRun  = int(opt.lastRun)
#calMode  = str(opt.calMode) if not str(opt.calMode)=='' else "StdBunch"
calMode  = str(opt.calMode) if not str(opt.calMode)=='' else "AagBunch" # Set default to AAG
MC=""
publish = (opt.publish=='True')
mail = "martin.delcourt@cern.ch "#dimattia@cern.ch"

usePCL = (opt.usePCL=='True')

initEnv='cd ' + os.getcwd() + ';'
initEnv+='source /afs/cern.ch/cms/cmsset_default.sh' + ';'
initEnv+='eval `scramv1 runtime -sh`' + ';'

name = "Run_"+str(firstRun)+"_to_"+str(lastRun)
if len(calMode)>0:  name = name+"_"+calMode
if(usePCL==True):   name = name+"_PCL"
else:               name = name+"_CalibTree"
print(name)

automatic = opt.automatic
scriptDir = opt.scriptDir


if(os.system(initEnv+"sh sequence.sh \"" + name + "\" \"" + calMode + "\" \"CMS Preliminary  -  Run " + str(firstRun) + " to " + str(lastRun) + "\"")!=0):
	os.system('echo "Gain calibration failed" | mail -s "Gain calibration failed ('+name+')" ' + mail)        
else:
	if(publish==True):os.system(initEnv+"sh sequence.sh " + name);
	os.system('echo "Manual gain calibration done\nhttps://test-stripcalibvalidation.web.cern.ch/test-stripcalibvalidation/CalibrationValidation/ParticleGain/" | mail -s "Gain calibration done ('+name+')" ' + mail)

if(automatic==True):
   #call the script one more time to make sure that we do not have a new run to process
   os.chdir(scriptDir); #go back to initial location
   os.system('python automatic_RunOnCalibTree.py')
