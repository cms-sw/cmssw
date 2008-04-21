import FWCore.ParameterSet.Config as cms

maxLayer1Taus = cms.EDFilter("PATTauMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Taus")
)


