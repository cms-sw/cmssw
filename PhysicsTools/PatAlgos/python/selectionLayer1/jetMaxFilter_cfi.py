import FWCore.ParameterSet.Config as cms

maxLayer1Jets = cms.EDFilter("PATJetMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Jets")
)


