import FWCore.ParameterSet.Config as cms

l1GctValidation = cms.EDFilter("L1GctValidation",
    rctInputTag  = cms.untracked.InputTag("simRctDigis"),
    gctInputTag  = cms.untracked.InputTag("simGctDigis")
)


