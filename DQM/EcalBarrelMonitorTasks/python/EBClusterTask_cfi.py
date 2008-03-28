import FWCore.ParameterSet.Config as cms

ecalBarrelClusterTask = cms.EDFilter("EBClusterTask",
    ClusterShapeAssociation = cms.InputTag("hybridSuperClusters","hybridShapeAssoc"),
    BasicClusterCollection = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    enableCleanup = cms.untracked.bool(True),
    SuperClusterCollection = cms.InputTag("hybridSuperClusters")
)


