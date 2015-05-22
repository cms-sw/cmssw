#!/bin/env python

import sys
import imp
import copy
import os
import shutil
import pickle
import math
import re

from CMGTools.Production.batchmanager import BatchManager

## def chunks(l, n):
##     return [l[i:i+n] for i in range(0, len(l), n)]

## def split(comps):
##     # import pdb; pdb.set_trace()
##     splitComps = []
##     for comp in comps:
##         if hasattr( comp, 'splitFactor') and comp.splitFactor>1:
##             chunkSize = len(comp.files) / comp.splitFactor
##             if len(comp.files) % comp.splitFactor:
##                 chunkSize += 1 
##             # print 'chunk size',chunkSize, len(comp.files), comp.splitFactor 
##             for ichunk, chunk in enumerate( chunks( comp.files, chunkSize)):
##                 newComp = copy.deepcopy(comp)
##                 newComp.files = chunk
##                 newComp.name = '{name}_Chunk{index}'.format(name=newComp.name,
##                                                        index=ichunk)
##                 splitComps.append( newComp )
##         else:
##             splitComps.append( comp )
##     return splitComps


def batchScriptCERN( index, remoteDir=''):
   '''prepare the LSF version of the batch script, to run on LSF'''
   script = """#!/bin/bash
#BSUB -q 8nm
echo 'environment:'
echo
env
ulimit -v 3000000
echo 'copying job dir to worker'
cd $CMSSW_BASE/src
eval `scram ru -sh`
cd -
cp -rf $LS_SUBCWD .
ls
cd `find . -type d | grep /`
echo 'running'
# here, run powheg
cat powheg.input > out.txt
echo 'sending the job directory back'
cp out.txt $LS_SUBCWD 
""" 
   return script


def batchScriptLocal(  remoteDir, index ):
   '''prepare a local version of the batch script, to run using nohup'''

   script = """#!/bin/bash
echo 'running'
cat powheg.input
echo
powheg powheg.input
echo 'sending the job directory back'
""" 
   return script


card_re = re.compile('(\S+)\s+(\d+).*')


class MyBatchManager( BatchManager ):
   '''Batch manager specific to cmsRun processes.''' 
         
   def PrepareJobUser(self, jobDir, jobValue ):
       '''Prepare one job. This function is called by the base class.'''
       print jobValue

       powheg_config = open(cfgFileName, 'r')
       ofile = open( '/'.join([jobDir, cfgFileName]), 'w')
       for line in powheg_config:
           # line = line.rstrip()
           
           match = card_re.match(line)
           if match:
               card_name = match.group(1)
               value = match.group(2)
               if card_name == 'iseed':
                   value = ''.join([value, str(jobValue)])
                   line = '\t'.join([card_name, value, '\n'])
           ofile.write(line)
       ofile.close()
       powheg_config

       #prepare the batch script
       scriptFileName = jobDir+'/batchScript.sh'
       scriptFile = open(scriptFileName,'w')
       # the line below is probably obsolete
       storeDir = self.remoteOutputDir_.replace('/castor/cern.ch/cms','')
       mode = self.RunningMode(options.batch)
       if mode == 'LXPLUS':
           scriptFile.write( batchScriptCERN( storeDir, value) )
       elif mode == 'LOCAL':
           scriptFile.write( batchScriptLocal( storeDir, value) ) 
       scriptFile.close()
       os.system('chmod +x %s' % scriptFileName)
       
##        shutil.copyfile(cfgFileName, jobDir+'/pycfg.py')
##        jobConfig = copy.deepcopy(config)
##        jobConfig.components = [ components[value] ]
##        cfgFile = open(jobDir+'/config.pck','w')
##        pickle.dump( jobConfig, cfgFile )
##        # pickle.dump( cfo, cfgFile )
##        cfgFile.close()

      
if __name__ == '__main__':
    batchManager = MyBatchManager()
    batchManager.parser_.usage="""
    %prog [options] <njobs> <cfgFile> 

    Run Colin's python analysis system on the batch.
    Job splitting is determined by your configuration file.
    """

    options, args = batchManager.ParseOptions()
    if len(args)!=2:
        print batchManager.parser_.usage
        print
        print 'need exactly two arguments'
        sys.exit(1)

    njobs, cfgFileName = args
    njobs = int(njobs)
    
    listOfValues = range(0, njobs)
    batchManager.PrepareJobs( listOfValues )
    waitingTime = 0.1
    batchManager.SubmitJobs( waitingTime )

    
    
##     cfo = imp.load_source("pycfg", cfgFileName, handle)
##     config = cfo.config
##     handle.close()

##     components = split( [comp for comp in config.components if len(comp.files)>0] )
##     listOfValues = range(0, len(components))
##     listOfNames = [comp.name for comp in components]

##     batchManager.PrepareJobs( listOfValues, listOfNames )
##     waitingTime = 0.1
##     batchManager.SubmitJobs( waitingTime )

