import FWCore.ParameterSet.Config as cms

digiMonitor = cms.EDFilter("DTDigiTask",
    tdcRescale = cms.untracked.int32(1),
    timeBoxGranularity = cms.untracked.int32(4),
    maxTDCHitsPerChamber = cms.untracked.int32(30000),
    readDB = cms.untracked.bool(True),
    defaultTtrig = cms.int32(2700),
    defaultTmax = cms.int32(500),
    checkNoisyChannels = cms.untracked.bool(True),
    inTimeHitsLowerBound = cms.int32(500),
    inTimeHitsUpperBound = cms.int32(500),
    enableMonitorDaemon = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),
    localrun = cms.untracked.bool(True),
    performPerWireT0Calibration = cms.bool(True)
)


