import FWCore.ParameterSet.Config as cms

TTStubsFromPhase2TrackerDigis = cms.EDProducer("TTStubBuilder_Phase2TrackerDigi_",
    TTClusters = cms.InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"),
    OnlyOnePerInputCluster = cms.bool(True)
)


