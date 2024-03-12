import FWCore.ParameterSet.Config as cms

egammaSuperClusterMerger = cms.EDProducer("SuperClusterMerger",
    src = cms.VInputTag(
        cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"), 
        cms.InputTag("correctedEndcapSuperClustersWithPreshower")
    )
)


# foo bar baz
# 6D5s6FnyMkOvv
# x5bXNAlF2OHkF
