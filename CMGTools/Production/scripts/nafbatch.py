#!/bin/env python

import sys
import imp
import copy
import os
import shutil
import pickle
import math

from CMGTools.Production.batchmanager import BatchManager
#from PhysicsTools.HeppyCore.utils.batchmanager import BatchManager
from PhysicsTools.HeppyCore.framework.heppy import split

def batchScriptNAF( jobDir='/nfs/dust/cms/user/lobanov/SUSY/Run2/CMG/CMSSW_7_0_6_patch1/src/CMGTools/TTHAnalysis/cfg/output_Directory/TTJets_PU20bx25_V52'):
   '''prepare the NAF version of the batch script, to run on NAF'''
   script = """#!/bin/bash
## make sure the right shell will be used
#$ -S /bin/zsh
## the cpu time for this job
#$ -l h_rt=02:59:00
## the maximum memory usage of this job
#$ -l h_vmem=1900M
## operating system
#$ -l distro=sld6
## architecture
##$ -l arch=amd64
## stderr and stdout are merged together to stdout
#$ -j y
##(send mail on job's end and abort)
##$ -m a
#$ -l site=hh
## transfer env var from submission host
#$ -V
## set cwd to submission host pwd
#$ -cwd
## define outputdir,executable,config file and LD_LIBRARY_PATH
#$ -v OUTDIR="""
#append Job directory
   script += jobDir
   script += """
## define dir for stdout
#$ -o """
#append log directory
   script += jobDir
   script += """/logs"""
   script += """
#start of script
echo job start at `date`
echo "Running on machine" `uname -a`
echo $(lsb_release -a | grep Description)
echo "Locating in" `pwd`

#cd $CMSSW_BASE/src
eval `/cvmfs/cms.cern.ch/common/scramv1 runtime -sh`
echo "CMSSW version:" $CMSSW_VERSION
echo "CMSSW base:" $CMSSW_BASE
echo "Python version" `python --version`

cd $OUTDIR
TaskID=$((SGE_TASK_ID+1))
#cd *_Chunk$TaskID
JobDir=$(find . -maxdepth 1 -type d ! -name "logs" | sed ''$TaskID'q;d')
echo "Changing to job dir" $JobDir
cd $JobDir

echo 'Running in dir' `pwd`
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck
echo
echo job end at `date`
"""
   return script

def batchScriptLocal(  remoteDir, index ):
   '''prepare a local version of the batch script, to run using nohup'''

   script = """#!/bin/bash
echo 'running'
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck
echo
echo 'sending the job directory back'
mv Loop/* ./
"""
   return script



class MyBatchManager( BatchManager ):
   '''Batch manager specific to cmsRun processes.'''

   def PrepareJobUser(self, jobDir, value ):
       '''Prepare one job. This function is called by the base class.'''
       print value
       print components[value]

       #prepare the batch script

       # array job requires only 1 batchscript
       if '-t' in options.batch:
           outputDir = self.outputDir_
           scriptFileName = outputDir+'/batchScript.sh'
       # batchscript in each jobDir
       else:
           outputDir = jobDir
           scriptFileName = jobDir+'/batchScript.sh'

       if not os.path.isfile(scriptFileName):
           scriptFile = open(scriptFileName,'w')
           storeDir = self.remoteOutputDir_.replace('/castor/cern.ch/cms','')
           print 'options are', options.batch
           mode = self.RunningMode(options.batch)

           if mode == 'LXPLUS':
               scriptFile.write( batchScriptCERN( storeDir, value) )
           elif mode == 'NAF':
               scriptFile.write( batchScriptNAF( outputDir ) )
           elif mode == 'LOCAL':
               scriptFile.write( batchScriptLocal( storeDir, value) )

           scriptFile.close()
           os.system('chmod +x %s' % scriptFileName)

       shutil.copyfile(cfgFileName, jobDir+'/pycfg.py')
#      jobConfig = copy.deepcopy(config)
#      jobConfig.components = [ components[value] ]
       cfgFile = open(jobDir+'/config.pck','w')
       pickle.dump(  components[value] , cfgFile )
       # pickle.dump( cfo, cfgFile )
       cfgFile.close()

   def SubmitJobArray(self, numbOfJobs):
       '''Submit all jobs as an array.'''
       outputDir = self.outputDir_
       print 'Number of jobs', numbOfJobs
       print 'Outputdir', outputDir

       self.mkdir(outputDir+"/logs")

       subline = self.options_.batch
       subline =  subline.replace("-t","-t 1-"+str(numbOfJobs))
       subline =  subline.replace("batchScript.sh",outputDir+"/batchScript.sh")
       print subline

       os.chdir( outputDir )
       os.system( subline )

if __name__ == '__main__':
    batchManager = MyBatchManager()
    batchManager.parser_.usage="""
    %prog [options] <cfgFile>

    Run Colin's python analysis system on the batch.
    Job splitting is determined by your configuration file.
    """

    options, args = batchManager.ParseOptions()

    cfgFileName = args[0]

    handle = open(cfgFileName, 'r')
    cfo = imp.load_source("pycfg", cfgFileName, handle)
    config = cfo.config
    handle.close()

    components = split( [comp for comp in config.components if len(comp.files)>0] )
    listOfValues = range(0, len(components))
    listOfNames = [comp.name for comp in components]

    print 'Preparing jobs'
    batchManager.PrepareJobs( listOfValues, listOfNames )

    if '-t' not in options.batch:
        waitingTime = 0.1
        batchManager.SubmitJobs( waitingTime )
    else:
        print 'Submitting job array'
        batchManager.SubmitJobArray(len(listOfNames))
