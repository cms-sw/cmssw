import FWCore.ParameterSet.Config as cms

hltFilteredLayerClustersCLUE3DHighL1Seeded = cms.EDProducer("FilteredLayerClustersProducer",
    LayerClusters = cms.InputTag("hltMergeLayerClustersL1Seeded"),
    LayerClustersInputMask = cms.InputTag("hltMergeLayerClustersL1Seeded","InitialLayerClustersMask"),
    clusterFilter = cms.string('ClusterFilterByAlgoAndSize'),
    iteration_label = cms.string('CLUE3DHigh'),
    max_cluster_size = cms.int32(9999),
    max_layerId = cms.int32(9999),
    mightGet = cms.optional.untracked.vstring,
    min_cluster_size = cms.int32(2),
    min_layerId = cms.int32(0)
)
