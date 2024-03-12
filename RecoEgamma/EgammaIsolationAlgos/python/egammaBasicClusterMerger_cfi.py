import FWCore.ParameterSet.Config as cms

egammaBasicClusterMerger = cms.EDProducer("BasicClusterMerger",
    src = cms.VInputTag(
        cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"), 
        cms.InputTag("islandBasicClusters","islandEndcapBasicClusters")
    )
)


# foo bar baz
# aPM5J7eDpAYvZ
# Kgk7PawqSneut
