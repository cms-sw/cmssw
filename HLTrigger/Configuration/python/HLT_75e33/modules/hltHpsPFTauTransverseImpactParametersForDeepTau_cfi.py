import FWCore.ParameterSet.Config as cms


hltHpsPFTauTransverseImpactParametersForDeepTau = cms.EDProducer( "PFTauTransverseImpactParameters",
    PFTauPVATag = cms.InputTag( "hltHpsPFTauPrimaryVertexProducerForDeepTau" ),
    useFullCalculation = cms.bool( True ),
    PFTauTag = cms.InputTag( "hltHpsPFTauProducer" ),
    PFTauSVATag = cms.InputTag( "hltHpsPFTauSecondaryVertexProducerForDeepTau" )
)
# foo bar baz
# FRXNR4KMtr2iP
