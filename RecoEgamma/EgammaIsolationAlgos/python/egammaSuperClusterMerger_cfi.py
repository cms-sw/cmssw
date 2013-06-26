import FWCore.ParameterSet.Config as cms

egammaSuperClusterMerger = cms.EDProducer("SuperClusterMerger",
    src = cms.VInputTag(
        cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"), 
        cms.InputTag("correctedEndcapSuperClustersWithPreshower")
    )
)


