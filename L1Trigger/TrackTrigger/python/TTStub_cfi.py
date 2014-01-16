import FWCore.ParameterSet.Config as cms

TTStubsFromPixelDigis = cms.EDProducer("TTStubBuilder_PixelDigi_",
    TTClusters = cms.InputTag("TTClustersFromPixelDigis", "ClusterInclusive"),
)


