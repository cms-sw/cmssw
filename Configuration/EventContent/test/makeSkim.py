#! /usr/bin/env python

r'''
cmsRun Configuration file that skims the data from the reconstructed events.
It is very general and allows to set in the metaconfig the parameters for the skimming.
'''

import FWCore.ParameterSet.Config as cms
# The meta configuration: 3 parameters
import metaconfig
print 'metaconfig.__dict__=%s'%metaconfig.__dict__

# The cff and cfi management
import fragments

process=cms.Process('RECOSIM') #The object that stores the configuration for cmsRun

includes_list=['FWCore/MessageLogger/data/MessageLogger.cfi',
               'Configuration/EventContent/data/EventContent.cff']

for el in fragments.include(includes_list):
    process.extend(el)
    
    
process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(metaconfig.nevts)) 

process.source=cms.Source('PoolSource',
                          fileNames=cms.untracked.vstring('file:%s' %metaconfig.infile))

# We use a feature of python to make general the content of the skimmed data..
exec('custom_outputCommands=process.%s.outputCommands\n'%metaconfig.outputCommands)

process.OutModule=cms.OutputModule('PoolOutputModule',
                                   fileName=cms.untracked.string('%s' %metaconfig.outfile),
                                   outputCommands=custom_outputCommands)

process.printEventNumber=cms.OutputModule('AsciiOutputModule') 

process.out=cms.EndPath(process.OutModule+process.printEventNumber)                                                       
  