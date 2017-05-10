import FWCore.ParameterSet.Config as cms


dqmStreamStats = cms.EDAnalyzer("DQMStreamStats",
    statsDepth = cms.untracked.int32(1),
    pathNameMatch = cms.untracked.string('*'),
    dumpMemoryHistory = cms.untracked.bool( True ),                             
    verbose = cms.untracked.int32(0),
    runInEventLoop = cms.untracked.bool(False),
    dumpOnEndLumi = cms.untracked.bool(True),
    dumpOnEndRun = cms.untracked.bool(True),
    runOnEndJob = cms.untracked.bool(False),
    dumpToFWJR = cms.untracked.bool(True)
)
