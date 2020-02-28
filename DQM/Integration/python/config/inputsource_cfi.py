from __future__ import print_function
from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import sys
from .dqmPythonTypes import *

options = VarParsing.VarParsing('analysis')

# options.inputFiles are inherited from 'analysis'
options.register('runNumber',
                 111,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number.")

options.register('runInputDir',
                 '/tmp',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Directory where the DQM files will appear.")

options.register('scanOnce',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Don't repeat file scans: use what was found during the initial scan. EOR file is ignored and the state is set to 'past end of run'.")

options.register('skipFirstLumis',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Skip (and ignore the minEventsPerLumi parameter) for the files which have been available at the begining of the processing. ")

# Parameters for runType

options.register ('runkey',
          'pp_run',
          VarParsing.VarParsing.multiplicity.singleton,
          VarParsing.VarParsing.varType.string,
          "Run Keys of CMS")

options.parseArguments()

# Fix to allow scram to compile
#if len(sys.argv) > 1:
#  options.parseArguments()

runType = RunType()
if not options.runkey.strip():
  options.runkey = 'pp_run'

runType.setRunType(options.runkey.strip())

if not options.inputFiles:
    # Input source
    nextLumiTimeoutMillis = 240000
    endOfRunKills = True
    
    if options.scanOnce:
        endOfRunKills = False
        nextLumiTimeoutMillis = 0
    
    source = cms.Source("DQMStreamerReader",
        runNumber = cms.untracked.uint32(options.runNumber),
        runInputDir = cms.untracked.string(options.runInputDir),
        SelectEvents = cms.untracked.vstring('*'),
        streamLabel = cms.untracked.string('streamDQM'),
        scanOnce = cms.untracked.bool(options.scanOnce),
        minEventsPerLumi = cms.untracked.int32(1),
        delayMillis = cms.untracked.uint32(500),
        nextLumiTimeoutMillis = cms.untracked.int32(nextLumiTimeoutMillis),
        skipFirstLumis = cms.untracked.bool(options.skipFirstLumis),
        deleteDatFiles = cms.untracked.bool(False),
        endOfRunKills  = cms.untracked.bool(endOfRunKills),
    )
else:
    print("The list of input files is provided. Disabling discovery and running on everything.")
    files = ["file://" + x for x in options.inputFiles]
    source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(files),
        secondaryFileNames = cms.untracked.vstring()
    )
    
#source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#       '/store/user/tosi/STEAM/DQM/online/outputDQM_3.root'
#    ),
#    secondaryFileNames = cms.untracked.vstring()
#)

print("Source:", source)
