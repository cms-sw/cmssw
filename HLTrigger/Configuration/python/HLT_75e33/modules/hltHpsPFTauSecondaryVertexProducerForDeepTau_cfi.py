import FWCore.ParameterSet.Config as cms

hltHpsPFTauSecondaryVertexProducerForDeepTau = cms.EDProducer( "PFTauSecondaryVertexProducer",
    PFTauTag = cms.InputTag( "hltHpsPFTauProducer" )
)
