import FWCore.ParameterSet.Config as cms

hltHpsPFTauSecondaryVertexProducer8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFTauSecondaryVertexProducer",
    PFTauTag = cms.InputTag("hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
    mightGet = cms.optional.untracked.vstring
)
