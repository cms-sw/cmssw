#!/usr/bin/env python
# Colin
# batch mode for cmsRun, March 2009


import os, sys,  imp
from optparse import OptionParser

from batchmanager import BatchManager
import castortools

import FWCore.ParameterSet.Config as cms


def batchScriptCCIN2P3():
   script = """#!/usr/local/bin/bash
#PBS -l platform=LINUX,u_sps_cmsf,M=2000MB,T=2000000
#PBS -q T
#PBS -eo
#PBS -me
#PBS -V

source $HOME/.bash_profile

echo '***********************'

ulimit -v 3000000

# coming back to submission dir do setup the env
cd $PBS_O_WORKDIR
eval `scramv1 ru -sh`


# back to the worker
cd -

# copy job dir here
cp -r $PBS_O_WORKDIR .

# go inside
jobdir=`ls`
echo $jobdir

cd $jobdir

cat > sysinfo.sh <<EOF
#! env bash
echo '************** ENVIRONMENT ****************'

env

echo
echo '************** WORKER *********************'
echo

free
cat /proc/cpuinfo 

echo
echo '************** START *********************'
echo
EOF

source sysinfo.sh > sysinfo.txt

cmsRun run_cfg.py

# copy job dir do disk
cd -
cp -r $jobdir $PBS_O_WORKDIR
"""
   return script


def batchScriptCERN( dir ):
   script = """#!/usr/local/bin/bash
#BSUB -q 8nm
echo 'environment:'
echo
env
ulimit -v 3000000
echo 'copying job dir to worker'
cd $LS_SUBCWD
eval `scramv1 ru -sh`
cd -
cp -rf $LS_SUBCWD .
ls
cd `find . -type d | grep /`
echo 'running'
cmsRun run_cfg.py
echo
echo 'sending the job directory back'
cp -rf * $LS_SUBCWD
#rm -rf $LS_SUBCWD
"""
   castorCopy = ''
   if dir != '':
      script += 'rfcp $LS_SUBCWD'
   
   return script


class MyBatchManager( BatchManager ):

    # prepare a job
    def PrepareJobUser(self, jobDir, value ):

       #prepare the batch script
       scriptFileName = jobDir+'/batchScript.sh'
       scriptFile = open(scriptFileName,'w')
       scriptFile.write( batchScriptCERN( self.options_.remoteOutputDir ) )
       scriptFile.close()
       os.system('chmod +x %s' % scriptFileName)

       #prepare the cfg
       process.source = fullSource.clone()
       # replace the list of fileNames by one of them

       iFileMin = (value-1)*grouping
       iFileMax = (value)*grouping

       print value, iFileMin, iFileMax

       process.source.fileNames = cms.untracked.vstring()

       for i in range(iFileMin, iFileMax):
          if(i>=len(fullSource.fileNames)):
             break
          process.source.fileNames.append( fullSource.fileNames[i] ) 

       print process.source
          
       cfgFile = open(jobDir+'/run_cfg.py','w')
       cfgFile.write(process.dumpPython())
       cfgFile.close()
       
    def SubmitJob( self, jobDir ):
       os.system('bsub < batchScript.sh')


batchManager = MyBatchManager()


batchManager.parser_.usage = "usage: %prog [options] grouping your_cfg.py"


(options,args) = batchManager.parser_.parse_args()
batchManager.ParseOptions()


if len(args)!=2:
   batchManager.parser_.print_help()
   sys.exit(1)

grouping = int(args[0])
cfgFileName = args[1]

# load cfg script
handle = open(cfgFileName, 'r')
cfo = imp.load_source("pycfg", cfgFileName, handle)
process = cfo.process
handle.close()

# keep track of the original source
fullSource = process.source.clone()
# will need to check the source contains local files
# if yes, do grouping.

print len(process.source.fileNames)
nJobs =  len(process.source.fileNames)/grouping
listOfValues = range( 0, nJobs)
print "range ", listOfValues
batchManager.PrepareJobs( listOfValues )

# if no, generate seeds

batchManager.SubmitJobs()




