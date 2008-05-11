import FWCore.ParameterSet.Config as cms

ecalBarrelClusterTask = cms.EDFilter("EBClusterTask",
    ClusterShapeAssociation = cms.InputTag("hybridSuperClusters","hybridShapeAssoc"),
    BasicClusterCollection = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    prefixME = cms.untracked.string('EcalBarrel'),
    SuperClusterCollection = cms.InputTag("hybridSuperClusters")
)


