import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    inputLabel = cms.InputTag("source"),
    gctFedId = cms.int32(745),
    hltMode = cms.bool(False),
    grenCompatibilityMode = cms.bool(False),
    verbose = cms.untracked.bool(False)
)


