import FWCore.ParameterSet.Config as cms

hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices = cms.EDFilter("PFTauSelector",
    cut = cms.string('pt > 15.00 & abs(eta) < 3.00'),
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag("hltHpsPFTauDiscriminationByTrackFinding8HitsMaxDeltaZWithOfflineVertices"),
            selectionCut = cms.double(0.5)
        ),
        cms.PSet(
            discriminator = cms.InputTag("hltHpsPFTauDiscriminationByTrackPt8HitsMaxDeltaZWithOfflineVertices"),
            selectionCut = cms.double(0.5)
        )
    ),
    src = cms.InputTag("hltHpsPFTaus8HitsMaxDeltaZWithOfflineVertices")
)
