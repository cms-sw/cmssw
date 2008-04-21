import FWCore.ParameterSet.Config as cms

minLayer1Taus = cms.EDFilter("PATTauMinFilter",
    src = cms.InputTag("selectedLayer1Taus"),
    minNumber = cms.uint32(0)
)


