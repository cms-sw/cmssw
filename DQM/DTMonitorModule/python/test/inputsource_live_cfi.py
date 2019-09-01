from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import sys

from DQM.Integration.config.dqmPythonTypes import RunType 
from DQM.DTMonitorModule.test.dtDqmPythonTypes import DTDQMConfig 

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

options.register('minEventsPerLumi',
                 1, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "If scanOnce is tue, sets the minimal # of events per LS to be processed before switching to the next LS (and file).")

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

# parameters for DT configuration

options.register('processAB7Digis',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable processing of AB7 DT digi data (duplicate occupancy plot and other customisations to DT digi monitoring)")

options.register('processAB7TPs',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable processing of AB7 DT local trigger data (DOES NOTHING FOR NOW)")

options.register('runWithLargeTimeBox',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Runs DTDigiTask with a timebox plot window of 6400 TDC Counts")

options.register('timeBoxTDCPedestal',
                 105100, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Pedestal to subtract to TDC counts for AB7")


options.parseArguments()

runType = RunType()
if not options.runkey.strip():
  options.runkey = 'pp_run'

dtDqmConfig = DTDQMConfig()
dtDqmConfig.setProcessAB7Digis(options.processAB7Digis)
dtDqmConfig.setProcessAB7TPs(options.processAB7TPs)

dtDqmConfig.setRunWithLargeTB(options.runWithLargeTimeBox)
dtDqmConfig.setTBTDCPedestal(options.timeBoxTDCPedestal)

runType.setRunType(options.runkey.strip())

if not options.inputFiles:
    # Input source
    nextLumiTimeoutMillis = 240000
    endOfRunKills = True
    
    if options.scanOnce:
        endOfRunKills = False
        nextLumiTimeoutMillis = 0
        minEventsPerLumi = options.minEventsPerLumi
    
    source = cms.Source("DQMStreamerReader",
        runNumber = cms.untracked.uint32(options.runNumber),
        runInputDir = cms.untracked.string(options.runInputDir),
        SelectEvents = cms.untracked.vstring('*'),
        streamLabel = cms.untracked.string('streamDQM'),
        scanOnce = cms.untracked.bool(options.scanOnce),
        minEventsPerLumi = cms.untracked.int32(minEventsPerLumi),
        delayMillis = cms.untracked.uint32(500),
        nextLumiTimeoutMillis = cms.untracked.int32(nextLumiTimeoutMillis),
        skipFirstLumis = cms.untracked.bool(options.skipFirstLumis),
        deleteDatFiles = cms.untracked.bool(False),
        endOfRunKills  = cms.untracked.bool(endOfRunKills),
    )
else:
    print("The list of input files is provided. Disabling discovery and running on everything.")
    files = map(lambda x: "file://" + x, options.inputFiles)
    source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(files),
        secondaryFileNames = cms.untracked.vstring()
    )

print("Source:", source)

print("RunType:", runType)

print("DTDQMConfig:", dtDqmConfig)

