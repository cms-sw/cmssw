import FWCore.ParameterSet.Config as cms

hltHpsPFTaus8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("RecoTauPiZeroUnembedder",
    src = cms.InputTag("hltHpsPFTauCleaner8HitsMaxDeltaZWithOfflineVertices")
)
