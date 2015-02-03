import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import sys
from dqmPythonTypes import *

options = VarParsing.VarParsing('analysis')

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

options.register('skipFirstLumis',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Skip (and ignore the minEventsPerLumi parameter) for the files which have been available at the begining of the processing. ")

# Parameters for runType

options.register('runtype',
         'cosmic_run',
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
          "Type of Run in CMS")

options.register ('runkey',
          'cosmic_run',
          VarParsing.VarParsing.multiplicity.singleton,
          VarParsing.VarParsing.varType.string,
          "Run Keys of CMS")

options.parseArguments()

# Fix to allow scram to compile
#if len(sys.argv) > 1:
#  options.parseArguments()

runType = RunType(['pp_run','cosmic_run','hi_run','hpu_run'])
if not options.runkey.strip():
  options.runkey = 'cosmic_run'

runType.setRunType(options.runkey.strip())

# Input source
source = cms.Source("DQMStreamerReader",
    runNumber = cms.untracked.uint32(options.runNumber),
    runInputDir = cms.untracked.string(options.runInputDir),
    SelectEvents = cms.untracked.vstring('*'),
    streamLabel = cms.untracked.string('streamDQM'),
    minEventsPerLumi = cms.untracked.int32(1),
    delayMillis = cms.untracked.uint32(500),
    nextLumiTimeoutMillis = cms.untracked.int32(30000),
    skipFirstLumis = cms.untracked.bool(options.skipFirstLumis),
    deleteDatFiles = cms.untracked.bool(False),
    endOfRunKills  = cms.untracked.bool(True),
)

