#!/bin/env python

import sys
import imp
import copy
import os
import shutil
import pickle
import math
from CMGTools.Production.batchmanager import BatchManager
from CMGTools.Production.datasetToSource import *

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def split(comps):
    # import pdb; pdb.set_trace()
    splitComps = []
    for comp in comps:
        if hasattr( comp, 'splitFactor') and comp.splitFactor>1:
            chunkSize = len(comp.files) / comp.splitFactor
            if len(comp.files) % comp.splitFactor:
                chunkSize += 1 
            # print 'chunk size',chunkSize, len(comp.files), comp.splitFactor 
            for ichunk, chunk in enumerate( chunks( comp.files, chunkSize)):
                newComp = copy.deepcopy(comp)
                newComp.files = chunk
                newComp.name = '{name}_Chunk{index}'.format(name=newComp.name,
                                                       index=ichunk)
                splitComps.append( newComp )
        else:
            splitComps.append( comp )
    return splitComps


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
eval `scramv1 ru -sh`
# cd $LS_SUBCWD
# eval `scramv1 ru -sh`
cd -
cp -rf $LS_SUBCWD .
ls
cd `find . -type d | grep /`
echo 'running'
python $CMSSW_BASE/src/CMGTools/RootTools/python/fwlite/Looper.py config.pck
echo
echo 'sending the job directory back'
cp -r Loop/* $LS_SUBCWD 
""" 
   return script


def batchScriptLocal( index ):
   '''prepare a local version of the batch script, to run using nohup'''

   script = """#!/bin/bash
echo 'running'
cmsRun run_cfg.py
""" 
   return script


def tuneProcess(process):
    # process.ZZ4muAnalysis.sample = "DYjets"
    # process.ZZ4eAnalysis.sample = "DYjets"
    # process.ZZ2mu2eAnalysis.sample = "DYjets"
    # process.p4 = cms.Path(~process.heavyflavorfilter)
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )


class MyBatchManager( BatchManager ):
   '''Batch manager specific to cmsRun processes.''' 
         
   def PrepareJobUser(self, jobDir, value ):
       '''Prepare one job. This function is called by the base class.'''
       print value
       print splitComponents[value]

       # import pdb; pdb.set_trace()
       #prepare the batch script
       scriptFileName = jobDir+'/batchScript.sh'
       scriptFile = open(scriptFileName,'w')
       # storeDir = self.remoteOutputDir_.replace('/castor/cern.ch/cms','')
       mode = self.RunningMode(options.batch)
       if mode == 'LXPLUS':
           scriptFile.write( batchScriptCERN( value ) )
       elif mode == 'LOCAL':
           scriptFile.write( batchScriptLocal( value ) ) 
       scriptFile.close()
       os.system('chmod +x %s' % scriptFileName)

       process.source.fileNames = splitComponents[value].files
       tuneProcess( process )

       cfgFile = open(jobDir+'/run_cfg.py','w')
       cfgFile.write('import FWCore.ParameterSet.Config as cms\n\n')
       # cfgFile.write('import os,sys\n')
       cfgFile.write( process.dumpPython() )
       cfgFile.write( '\n' )
       cfgFile.close()


class Component(object):

    def __init__(self, name, user, dataset, splitFactor ):
        self.name = name
        self.files = datasetToSource( user, dataset ).fileNames
        self.splitFactor = splitFactor

        
      
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
    process = cfo.process
    components = [ Component(na, us, da, sp) for na, us,da,sp in cfo.components ]
    handle.close()


    splitComponents = split( components )
    listOfValues = range(0, len(splitComponents))
    listOfNames = [comp.name for comp in splitComponents]

    batchManager.PrepareJobs( listOfValues, listOfNames )

    waitingTime = 0.1
    batchManager.SubmitJobs( waitingTime )

