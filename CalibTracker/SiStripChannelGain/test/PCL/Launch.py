#!/usr/bin/env python3

import urllib
import string
import os
import sys
import subprocess

DATASET="/MinimumBias/Run2012C-SiStripCalMinBias-v2/ALCARECO"
FIRSTRUN=0 #200190

runs = []
results = sorted(subprocess.getstatusoutput('dbs search --query="find run,sum(block.numevents) where dataset='+DATASET+' and run>='+str(FIRSTRUN)+'"')[1].splitlines())
for line in results:
   linesplit = line.split('   ')
   if(len(linesplit)<2):continue
   run     = int(line.split('   ')[0])
   NEvents = int(line.split('   ')[1])
   if(NEvents>100000): runs.append(run)



subprocess.getstatusoutput('mkdir -p runs')
for r in runs:
    initEnv=''
    initEnv+='cd ' + os.getcwd() + '/runs/'+str(r)+'/;'
    initEnv+='source /afs/cern.ch/cms/cmsset_default.sh' + ';'
    initEnv+='eval `scramv1 runtime -sh`;'
    subprocess.getstatusoutput('mkdir -p runs/'+str(r))
    print("submitting jobs for run " + str(r))
    config_file=open('runs/'+str(r)+'/cmsDriver.sh','w')
    config_file.write( initEnv + 'cmsDriver.py run'+str(r)+' --datatier ALCARECO --conditions auto:com10 -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n -1 --dasquery=\'file dataset=/MinimumBias/Run2012C-SiStripCalMinBias-v2/ALCARECO run='+str(r)+'\'  --fileout file:run'+str(r)+'_out.root')
    config_file.close()
    out = subprocess.getoutput('bsub -q 2nw -J gainPCLrun' + str(r) +' "sh ' +  os.getcwd() + '/runs/'+str(r)+'/cmsDriver.sh"')
#    print('bsub -q 2nw -J gainPCLrun' + str(r) +' " ' + initEnv + 'cmsDriver.py run'+str(r)+' --datatier ALCARECO --conditions auto:com10 -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n -1 --dasquery=\'file dataset=/MinimumBias/Run2012C-SiStripCalMinBias-v2/ALCARECO run='+str(r)+'\'  --fileout file:run'+str(r)+'_out.root "')



