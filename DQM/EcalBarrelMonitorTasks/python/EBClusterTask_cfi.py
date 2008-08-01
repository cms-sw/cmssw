import FWCore.ParameterSet.Config as cms

ecalBarrelClusterTask = cms.EDFilter("EBClusterTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    BasicClusterCollection = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    SuperClusterCollection = cms.InputTag("hybridSuperClusters"),
    ClusterShapeAssociation = cms.InputTag("hybridSuperClusters","hybridShapeAssoc")
)

