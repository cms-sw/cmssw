import FWCore.ParameterSet.Config as cms

# module to filter on the maximal number of Taus
maxLayer1Taus = cms.EDFilter("PATTauMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Taus")
)


