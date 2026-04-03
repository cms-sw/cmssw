import FWCore.ParameterSet.Config as cms

hltFilteredLayerClustersCLUE3DBarrel = cms.EDProducer("FilteredLayerClustersProducer",
    LayerClusters = cms.InputTag("hltMergeLayerClusters"),
    LayerClustersInputMask = cms.InputTag("hltMergeLayerClusters", "InitialLayerClustersMask"),
    clusterFilter = cms.string('ClusterFilterByAlgo'),
    iteration_label = cms.string('CLUE3DBarrel'),
    max_cluster_size = cms.int32(9999),
    max_layerId = cms.int32(9999),
    min_cluster_size = cms.int32(0),
    min_layerId = cms.int32(0),
    algo_number = cms.vint32(10, 11)
)
