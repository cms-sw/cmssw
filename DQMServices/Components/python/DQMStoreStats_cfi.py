import FWCore.ParameterSet.Config as cms


dqmStoreStats = cms.EDFilter("DQMStoreStats",
    statsDepth = cms.untracked.int32(1),
    pathNameMatch = cms.untracked.string('*'),
    verbose = cms.untracked.int32(0),
    runInEventLoop = cms.untracked.bool(False),
    runOnEndLumi = cms.untracked.bool(False),
    runOnEndRun = cms.untracked.bool(True),
    runOnEndJob = cms.untracked.bool(False)
)
