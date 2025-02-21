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

unitTest = 'unitTest=True' in sys.argv

#-----------------------------
if unitTest:
  process.load("DQM.Integration.config.unitteststreamerinputsource_cfi")
  from DQM.Integration.config.unitteststreamerinputsource_cfi import options
else:
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options

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
# process.dqmSaverPB.tag         = 'OnlineBeamMonitor'
# process.dqmSaverPB.runNumber   = options.runNumber

# for running offline enhance the time validity of the online beamspot in DB
if (unitTest or process.isDqmPlayback.value):
  process.BeamSpotESProducer.timeThreshold = cms.int32(int(1e6))

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
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

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
                               * process.dqmSaver )#* process.dqmSaverPB)

process.monitor = cms.Sequence(process.dqmOnlineBeamMonitor)

#-----------------------------------------------------------
# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

process.p = cms.Path( process.dqmcommon * process.monitor )

print("Final Source settings:", process.source)
