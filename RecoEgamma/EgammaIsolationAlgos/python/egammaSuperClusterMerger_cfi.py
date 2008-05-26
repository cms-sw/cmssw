import FWCore.ParameterSet.Config as cms

egammaSuperClusterMerger = cms.EDFilter("SuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"), cms.InputTag("correctedEndcapSuperClustersWithPreshower"))
)


