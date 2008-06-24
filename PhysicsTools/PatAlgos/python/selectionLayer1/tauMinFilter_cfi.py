import FWCore.ParameterSet.Config as cms

# module to filter on the minimal number of Taus
minLayer1Taus = cms.EDFilter("PATTauMinFilter",
    src = cms.InputTag("selectedLayer1Taus"),
    minNumber = cms.uint32(0)
)


