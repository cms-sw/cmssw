import FWCore.ParameterSet.Config as cms


hltHpsPFTauTransverseImpactParametersForDeepTau = cms.EDProducer( "PFTauTransverseImpactParameters",
    PFTauPVATag = cms.InputTag( "hltHpsPFTauPrimaryVertexProducerForDeepTau" ),
    useFullCalculation = cms.bool( True ),
    PFTauTag = cms.InputTag( "hltHpsPFTauProducer" ),
    PFTauSVATag = cms.InputTag( "hltHpsPFTauSecondaryVertexProducerForDeepTau" )
)
