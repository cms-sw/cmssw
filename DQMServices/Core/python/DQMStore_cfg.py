import FWCore.ParameterSet.Config as cms

DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    verboseQT = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False)
)
