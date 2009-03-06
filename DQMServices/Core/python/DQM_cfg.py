import FWCore.ParameterSet.Config as cms

DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False)
)

DQM = cms.Service("DQM",
    debug = cms.untracked.bool(False),
    publishFrequency = cms.untracked.double(5.0),
    collectorPort = cms.untracked.int32(9090),
    collectorHost = cms.untracked.string('localhost'),
    filter = cms.untracked.string('')
)
