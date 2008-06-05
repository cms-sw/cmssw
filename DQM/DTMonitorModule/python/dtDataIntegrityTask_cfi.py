import FWCore.ParameterSet.Config as cms

DTDataIntegrityTask = cms.Service("DTDataIntegrityTask",
    TBhistoGranularity = cms.untracked.int32(1),
    getSCInfo = cms.untracked.bool(True),
    debug = cms.untracked.bool(False),
    doTimeHisto = cms.untracked.bool(True),
    timeBoxLowerBound = cms.untracked.int32(0),
    resetCycle = cms.untracked.int32(1000),
    timeBoxUpperBound = cms.untracked.int32(10000)
)


