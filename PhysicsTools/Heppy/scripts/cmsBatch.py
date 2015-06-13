#!/usr/bin/env python
# Colin
# batch mode for cmsRun, March 2009

import os, sys,  imp, re, pprint, string, time,shutil,copy,pickle,math
from optparse import OptionParser

# particle flow specific
from PhysicsTools.HeppyCore.utils.batchmanager import BatchManager
import PhysicsTools.HeppyCore.utils.eostools as castortools

# cms specific
import FWCore.ParameterSet.Config as cms
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper




def batchScriptCCIN2P3():
   script = """!/usr/bin/env bash
#PBS -l platform=LINUX,u_sps_cmsf,M=2000MB,T=2000000
# sets the queue
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


def batchScriptCERN(  remoteDir, index ):
   '''prepare the LSF version of the batch script, to run on LSF'''
   script = """#!/bin/bash
# sets the queue
#BSUB -q 8nm

echo 'environment:'
echo
env
ulimit -v 3000000
echo 'copying job dir to worker'
cd $CMSSW_BASE/src
eval `scramv1 ru -sh`
cd -
cp -rf $LS_SUBCWD .
ls
cd `find . -type d | grep /`
echo 'running'
%s run_cfg.py
if [ $? != 0 ]; then
    echo wrong exit code! removing all root files
    rm *.root
    exit 1 
fi
echo 'sending the job directory back'
""" % prog

   if remoteDir != '':
      remoteDir = castortools.eosToLFN(remoteDir) #remoteDir.replace('/eos/cms','')
      script += """
for file in *.root; do
newFileName=`echo $file | sed -r -e 's/\./_%s\./'`
fullFileName=%s/$newFileName
#this does cmsStage, but with retries
cmsStageWithFailover.py -f $file $fullFileName
#write the files as user readable but not writable
eos chmod 755 /eos/cms/$fullFileName
done
""" % (index, remoteDir)         
      script += 'rm *.root\n'
   script += 'cp -rf * $LS_SUBCWD\n'
   
   return script


def batchScriptLocal(  remoteDir, index ):
   '''prepare a local version of the batch script, to run using nohup'''

   script = """#!/bin/bash
echo 'running'
%s run_cfg.py
if [ $? != 0 ]; then
    echo wrong exit code! removing all root files
    rm *.root
    exit 1 
fi
echo 'sending the job directory back'
""" % prog

   if remoteDir != '':
      remoteDir = castortools.eosToLFN(remoteDir)
      script += """
for file in *.root; do
newFileName=`echo $file | sed -r -e 's/\./_%s\./'`
cmsStageWithFailover.py -f $file $fullFileName
eos chmod 755 /eos/cms/$fullFileName
done
""" % (index, remoteDir)
      script += 'rm *.root\n'
   return script


class CmsBatchException( Exception):
   '''Exception class for this script'''
   
   def __init__(self, value):
      self.value = value
      
   def __str__(self):
      return str( self.value)


class MyBatchManager( BatchManager ):
   '''Batch manager specific to cmsRun processes.''' 

   def PrepareJobUser(self, jobDir, value ):
      '''Prepare one job. This function is called by the base class.'''
      
      process.source = fullSource.clone()
      
      #prepare the batch script
      scriptFileName = jobDir+'/batchScript.sh'
      scriptFile = open(scriptFileName,'w')
      storeDir = self.remoteOutputDir_.replace('/castor/cern.ch/cms','')
      mode = self.RunningMode(options.batch)
      if mode == 'LXPLUS':
         scriptFile.write( batchScriptCERN( storeDir, value) )    #here is the call to batchScriptCERN, i need to change value
      elif mode == 'LOCAL':
         scriptFile.write( batchScriptLocal( storeDir, value) )   #same as above but for batchScriptLocal
      scriptFile.close()
      os.system('chmod +x %s' % scriptFileName)

      #prepare the cfg
      # replace the list of fileNames by a chunk of filenames:
      if generator:
         randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
         randSvc.populate()
      else:
         iFileMin = (value-1)*grouping 
         iFileMax = (value)*grouping 
         process.source.fileNames = fullSource.fileNames[iFileMin:iFileMax]
         print process.source
      cfgFile = open(jobDir+'/run_cfg.py','w')
      cfgFile.write('import FWCore.ParameterSet.Config as cms\n\n')
      cfgFile.write('import os,sys\n')
      # need to import most of the config from the base directory containing all jobs
      cfgFile.write("sys.path.append('%s')\n" % os.path.dirname(jobDir) )
      cfgFile.write('from base_cfg import *\n')
      cfgFile.write('process.source = ' + process.source.dumpPython() + '\n')
      if generator:
         cfgFile.write('process.RandomNumberGeneratorService = ' + process.RandomNumberGeneratorService.dumpPython() + '\n')
      cfgFile.close()


batchManager = MyBatchManager()


file = open('cmsBatch.txt', 'w')
file.write(string.join(sys.argv) + "\n")
file.close()

batchManager.parser_.usage = """
%prog [options] <number of input files per job> <your_cfg.py>.

Submits a number of jobs taking your_cfg.py as a template. your_cfg.py can either read events from input files, or produce them with a generator. In the later case, the seeds are of course updated for each job.

A local output directory is created locally. This directory contains a job directory for each job, and a Logger/ directory containing information on the software you are using. 
By default:
- the name of the output directory is created automatically.
- the output root files end up in the job directories.

Each job directory contains:
- the full python configuration for this job. You can run it interactively by doing:
cmsRun run_cfg.py
- the batch script to run the job. You can submit it again by calling the batch command yourself, see the -b option.
- while running interactively: nohup.out, where the job stderr and stdout are redirected. To check the status of a job running interactively, do:
tail nohup.out
- after running:
  o the full nohup.out (your log) and your root files, in case you ran interactively
  o the LSF directory, in case you ran on LSF

Also see fwBatch.py, which is a layer on top of cmsBatch.py adapted to the organization of our samples on the CMST3. 

Examples:

First do:
cd $CMSSW_BASE/src/CMGTools/Common/test

to run on your local machine:
cmsBatch.py 1 testCMGTools_cfg.py -b 'nohup ./batchScript.sh&' 

to run on LSF (you must be logged on lxplus, not on your interactive machine, so that you have access to LSF)
cmsBatch.py 1 testCMGTools_cfg.py -b 'bsub -q 8nm < ./batchScript.sh' 
"""
batchManager.parser_.add_option("-p", "--program", dest="prog",
                                help="program to run on your cfg file",
                                default="cmsRun")
## batchManager.parser_.add_option("-b", "--batch", dest="batch",
##                                 help="batch command. default is: 'bsub -q 8nh < batchScript.sh'. You can also use 'nohup < ./batchScript.sh &' to run locally.",
##                                 default="bsub -q 8nh < .batchScript.sh")
batchManager.parser_.add_option("-c", "--command-args", dest="cmdargs",
                                help="command line arguments for the job",
                                default=None)
batchManager.parser_.add_option("--notagCVS", dest="tagPackages",
                                default=True,action="store_false",
                                help="tag the package on CVS (True)")

(options,args) = batchManager.parser_.parse_args()
batchManager.ParseOptions()

prog = options.prog
doCVSTag = options.tagPackages

if len(args)!=2:
   batchManager.parser_.print_help()
   sys.exit(1)

# testing that we run a sensible batch command. If not, exit.
runningMode = None
try:
   runningMode = batchManager.RunningMode( options.batch )
except CmsBatchException as err:
   print err
   sys.exit(1)

grouping = int(args[0])
nJobs = grouping
cfgFileName = args[1]

print 'Loading cfg'

pycfg_params = options.cmdargs
trueArgv = sys.argv
sys.argv = [cfgFileName]
if pycfg_params:
   sys.argv.extend(pycfg_params.split(' '))
print  sys.argv


# load cfg script
handle = open(cfgFileName, 'r')
cfo = imp.load_source("pycfg", cfgFileName, handle)
process = cfo.process
handle.close()

# Restore original sys.argv
sys.argv = trueArgv


# keep track of the original source
fullSource = process.source.clone()
generator = False

try:
   process.source.fileNames
except:
   print 'No input file. This is a generator process.'
   generator = True
   listOfValues = [i+1 for i in range( nJobs )] #Here is where the list of values is created 
else:
   print "Number of files in the source:",len(process.source.fileNames), ":"
   pprint.pprint(process.source.fileNames)
   nFiles = len(process.source.fileNames)
   nJobs = nFiles / grouping
   if (nJobs!=0 and (nFiles % grouping) > 0) or nJobs==0:
      nJobs = nJobs + 1
      
   print "number of jobs to be created: ", nJobs
   listOfValues = [i+1 for i in range( nJobs )] #OR Here is where the list of values is created
   #here i change from e.g 0-19 to 1-20

batchManager.PrepareJobs( listOfValues ) #PrepareJobs with listOfValues as param

# preparing master cfg file

cfgFile = open(batchManager.outputDir_+'/base_cfg.py','w')
cfgFile.write( process.dumpPython() + '\n')
cfgFile.close()

# need to wait 5 seconds to give castor some time
# now on EOS, should be ok. reducing to 1 sec
waitingTime = 1
if runningMode == 'LOCAL':
   # of course, not the case when running with nohup
   # because we will never have enough processes to saturate castor.
   waitingTime = 0
batchManager.SubmitJobs( waitingTime )


# logging

from PhysicsTools.HeppyCore.utils.logger import logger

oldPwd = os.getcwd()
os.chdir(batchManager.outputDir_)
logDir = 'Logger'
os.system( 'mkdir ' + logDir )
log = logger( logDir )

log.logCMSSW()
log.logJobs(nJobs)
#COLIN not so elegant... but tar is behaving in a strange way.
log.addFile( oldPwd + '/' + cfgFileName )

if not batchManager.options_.negate:
   if batchManager.remoteOutputDir_ != "":
      # we don't want to crush an existing log file on castor
      #COLIN could protect the logger against that.
      log.stageOut( batchManager.remoteOutputDir_ )
      
os.chdir( oldPwd )


