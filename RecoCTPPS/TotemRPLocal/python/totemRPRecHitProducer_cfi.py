import FWCore.ParameterSet.Config as cms

totemRPRecHitProducer = cms.EDProducer("TotemRPRecHitProducer",
    verbosity = cms.int32(0),
    tagCluster = cms.InputTag("totemRPClusterProducer")
)
