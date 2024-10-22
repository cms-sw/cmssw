import FWCore.ParameterSet.Config as cms

ThroughputService = cms.Service("ThroughputService",
    enableDQM = cms.untracked.bool(False),
    eventRange = cms.untracked.uint32(1000),
    eventResolution = cms.untracked.uint32(50),
    printEventSummary = cms.untracked.bool(True)
)
