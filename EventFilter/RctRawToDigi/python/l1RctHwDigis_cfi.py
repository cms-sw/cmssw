import FWCore.ParameterSet.Config as cms

l1RctHwDigis = cms.EDProducer("RctRawToDigi",
    inputLabel = cms.InputTag("source"),
    rctFedId = cms.untracked.int32(1350),
    verbose = cms.untracked.bool(False)
)
