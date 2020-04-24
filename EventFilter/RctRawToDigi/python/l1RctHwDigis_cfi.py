import FWCore.ParameterSet.Config as cms

rctDigis  = cms.EDProducer("RctRawToDigi",
    inputLabel = cms.InputTag("rawDataCollector"),
    rctFedId = cms.untracked.int32(1350),
    verbose = cms.untracked.bool(False)
)
