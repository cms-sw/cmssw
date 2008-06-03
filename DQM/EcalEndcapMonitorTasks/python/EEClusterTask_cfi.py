import FWCore.ParameterSet.Config as cms

ecalEndcapClusterTask = cms.EDFilter("EEClusterTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    BasicClusterCollection = cms.InputTag("islandBasicClusters","islandEndcapBasicClusters"),
    SuperClusterCollection = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters"),
    ClusterShapeAssociation = cms.InputTag("islandBasicClusters","islandEndcapShapeAssoc")
)

