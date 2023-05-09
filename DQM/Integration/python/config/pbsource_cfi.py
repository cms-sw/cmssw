from __future__ import print_function
from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import sys
from .dqmPythonTypes import *

options = VarParsing.VarParsing('analysis')

options.register('runNumber',
                 111,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number.")

options.register('datafnPosition',
                 3, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Data filename position in the positional arguments array 'data' in json file.")

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
                 "Skip (and ignore the minEventsPerLumi parameter) for the files which have been available at the beginning of the processing.")

options.register('BeamSplashRun',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Set client source settings for beam SPLASH run")

# Parameters for runType

options.register('runkey',
                 'pp_run',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Run Keys of CMS")

# Parameter for frontierKey

options.register('runUniqueKey',
                 'InValid',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Unique run key from RCMS for Frontier")

options.parseArguments()

# Fix to allow scram to compile
#if len(sys.argv) > 1:
#  options.parseArguments()

runType = RunType()
if not options.runkey.strip():
  options.runkey = 'pp_run'

runType.setRunType(options.runkey.strip())

# Input source
nextLumiTimeoutMillis = 120000
endOfRunKills = True

if options.scanOnce:
    endOfRunKills = False
    nextLumiTimeoutMillis = 0

source = cms.Source("DQMProtobufReader",
    runNumber = cms.untracked.uint32(options.runNumber),
    runInputDir = cms.untracked.string(options.runInputDir),
    streamLabel = cms.untracked.string('streamDQMHistograms'),
    scanOnce = cms.untracked.bool(options.scanOnce),
    datafnPosition = cms.untracked.uint32(options.datafnPosition),
    delayMillis = cms.untracked.uint32(500),
    nextLumiTimeoutMillis = cms.untracked.int32(nextLumiTimeoutMillis),
    skipFirstLumis = cms.untracked.bool(options.skipFirstLumis),
    deleteDatFiles = cms.untracked.bool(False),
    loadFiles = cms.untracked.bool(True),
    endOfRunKills  = cms.untracked.bool(endOfRunKills),
)

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/CMSBeamSplash2017
def set_BeamSplashRun_settings( source ):
  # source.minEventsPerLumi      = 1000000
  source.nextLumiTimeoutMillis = 15000

if options.BeamSplashRun : set_BeamSplashRun_settings( source )

print("Initial Source settings:", source)
