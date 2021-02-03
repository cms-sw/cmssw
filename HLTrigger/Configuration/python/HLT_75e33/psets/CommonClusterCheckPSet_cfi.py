import FWCore.ParameterSet.Config as cms

CommonClusterCheckPSet = cms.PSet(
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(1000000),
    MaxNumberOfPixelClusters = cms.uint32(100000),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    doClusterCheck = cms.bool(True)
)