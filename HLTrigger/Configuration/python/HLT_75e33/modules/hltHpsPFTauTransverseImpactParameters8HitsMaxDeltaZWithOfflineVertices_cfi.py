import FWCore.ParameterSet.Config as cms

hltHpsPFTauTransverseImpactParameters8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFTauTransverseImpactParameters",
    PFTauPVATag = cms.InputTag("hltHpsPFTauPrimaryVertexProducer8HitsMaxDeltaZWithOfflineVertices"),
    PFTauSVATag = cms.InputTag("hltHpsPFTauSecondaryVertexProducer8HitsMaxDeltaZWithOfflineVertices"),
    PFTauTag = cms.InputTag("hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
    mightGet = cms.optional.untracked.vstring,
    useFullCalculation = cms.bool(True)
)
