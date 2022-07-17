import FWCore.ParameterSet.Config as cms

filteredLayerClustersEM = cms.EDProducer("FilteredLayerClustersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    LayerClustersInputMask = cms.InputTag("ticlTrackstersTrkEM"),
    algo_number = cms.int32(8),
    clusterFilter = cms.string('ClusterFilterByAlgoAndSizeAndLayerRange'),
    iteration_label = cms.string('EM'),
    max_cluster_size = cms.int32(9999),
    max_layerId = cms.int32(30),
    mightGet = cms.optional.untracked.vstring,
    min_cluster_size = cms.int32(3),
    min_layerId = cms.int32(0)
)
