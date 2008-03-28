import FWCore.ParameterSet.Config as cms

ecalEndcapClusterTask = cms.EDFilter("EEClusterTask",
    ClusterShapeAssociation = cms.InputTag("islandBasicClusters","islandEndcapShapeAssoc"),
    BasicClusterCollection = cms.InputTag("islandBasicClusters","islandEndcapBasicClusters"),
    enableCleanup = cms.untracked.bool(True),
    SuperClusterCollection = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters")
)


