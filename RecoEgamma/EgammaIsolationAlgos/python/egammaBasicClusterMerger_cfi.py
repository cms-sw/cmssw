import FWCore.ParameterSet.Config as cms

egammaBasicClusterMerger = cms.EDFilter("BasicClusterMerger",
    src = cms.VInputTag(cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"), cms.InputTag("islandBasicClusters","islandEndcapBasicClusters"))
)


