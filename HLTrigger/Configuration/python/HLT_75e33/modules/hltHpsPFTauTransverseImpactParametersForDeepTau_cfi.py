import FWCore.ParameterSet.Config as cms

hltHpsPFTauTransverseImpactParametersForDeepTau = cms.EDProducer("PFTauTransverseImpactParameters",
    PFTauPVATag = cms.InputTag("hltHpsPFTauPrimaryVertexProducerForDeepTau"),
    PFTauSVATag = cms.InputTag("hltHpsPFTauSecondaryVertexProducerForDeepTau"),
    PFTauTag = cms.InputTag("hltHpsPFTauProducer"),
    useFullCalculation = cms.bool(True)
)
