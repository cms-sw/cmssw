import FWCore.ParameterSet.Config as cms

hltHpsPFTauProducer = cms.EDProducer( "RecoTauPiZeroUnembedder",
    src = cms.InputTag( "hltHpsPFTauProducerSansRefs" )
)
