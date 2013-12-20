import FWCore.ParameterSet.Config as cms

hpsPFTauProducer = cms.EDProducer(
            "RecoTauPiZeroUnembedder",
            src = cms.InputTag("hpsPFTauProducerSansRefs")
        )
