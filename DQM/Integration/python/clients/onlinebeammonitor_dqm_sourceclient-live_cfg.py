from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# Define once the BeamSpotOnline record name,
# will be used both in BeamMonitor setup and in payload creation/upload

import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("OnlineBeamMonitor", Run3)

# Message logger
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    cerr = cms.untracked.PSet(
#        FwkReport = cms.untracked.PSet(
#            optionalPSet = cms.untracked.bool(True),
#            reportEvery = cms.untracked.int32(1000),
#            limit = cms.untracked.int32(999999)
#        )
#    ),
#    destinations = cms.untracked.vstring('cerr'),
#)


unitTest=False
if 'unitTest=True' in sys.argv:
  unitTest=True
#-----------------------------
if unitTest:
  import FWCore.ParameterSet.VarParsing as VarParsing
  options = VarParsing.VarParsing("analysis")

  options.register(
      "runkey",
      "pp_run",
      VarParsing.VarParsing.multiplicity.singleton,
      VarParsing.VarParsing.varType.string,
      "Run Keys of CMS"
  )

  options.register('runNumber',
                  346508,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Run number. This run number has to be present in the dataset configured with the dataset option.")

  options.register('dataset',
                  '/ExpressPhysics/Commissioning2021-Express-v1/FEVT',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Dataset name like '/ExpressCosmics/Commissioning2019-Express-v1/FEVT'")

  options.register('maxLumi',
                  2,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Only lumisections up to maxLumi are processed.")

  options.register('minLumi',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Only lumisections starting from minLumi are processed.")

  options.register('lumiPattern',
                  '*',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Only lumisections with numbers matching lumiPattern are processed.")

  options.register('eventsPerLumi',
                  100,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "This number of last events in each lumisection will be processed.")

  # This is used only by the online clients themselves. 
  # We need to register it here because otherwise an error occurs saying that there is an unidentified option.
  options.register('unitTest',
                  True,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "Required to avoid the error.")

  options.register('noDB',
                  True, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "Don't upload the BeamSpot conditions to the DB")

  options.parseArguments()
  
  process.source = cms.Source("EmptySource")
  process.source.numberEventsInRun=cms.untracked.uint32(100)
  process.source.firstRun = cms.untracked.uint32(options.runNumber)
  process.source.firstLuminosityBlock = cms.untracked.uint32(1)
  process.source.numberEventsInLuminosityBlock = cms.untracked.uint32(2)
  process.maxEvents = cms.untracked.PSet(
              input = cms.untracked.int32(100)
)

else:
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options
  # for live online DQM in P5
  # new stream label
  #process.source.streamLabel = cms.untracked.string('streamDQMOnlineBeamspot')

#ESProducer
process.load("CondCore.CondDB.CondDB_cfi")
process.BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")

#-----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'OnlineBeamMonitor'
process.dqmSaver.tag           = 'OnlineBeamMonitor'
process.dqmSaver.runNumber     = options.runNumber
process.dqmSaverPB.tag         = 'OnlineBeamMonitor'
process.dqmSaverPB.runNumber   = options.runNumber

#-----------------------------
# BeamMonitor
#-----------------------------
process.dqmOnlineBeamMonitor = cms.EDProducer("OnlineBeamMonitor",
MonitorName         = cms.untracked.string("OnlineBeamMonitor"),
AppendRunToFileName = cms.untracked.bool(False),
WriteDIPAscii       = cms.untracked.bool(True),
DIPFileName         = cms.untracked.string("/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResultsForDIP.txt")
)

#---------------
# Calibration
#---------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

# Please *do not* delete this toGet statement as it is needed to fetch BeamSpotOnline
# information every lumisection (instead of every run as for the other records in the GT)
process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(
    record = cms.string("BeamSpotOnlineLegacyObjectsRcd"),
    refreshTime = cms.uint64(2)
  ),
  cms.PSet(
    record = cms.string("BeamSpotOnlineHLTObjectsRcd"),
    refreshTime = cms.uint64(2)
  )
)

process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver * process.dqmSaverPB)

process.monitor = cms.Sequence(process.dqmOnlineBeamMonitor)

#-----------------------------------------------------------
# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)


process.p = cms.Path( process.dqmcommon
                        * process.monitor )
print("Final Source settings:", process.source)
