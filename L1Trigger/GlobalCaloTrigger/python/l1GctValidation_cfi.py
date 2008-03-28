import FWCore.ParameterSet.Config as cms

l1GctValidation = cms.EDFilter("L1GctValidation",
    inputTag = cms.untracked.InputTag("gctDigis")
)


