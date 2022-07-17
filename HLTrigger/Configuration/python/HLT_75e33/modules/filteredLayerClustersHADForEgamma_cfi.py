import FWCore.ParameterSet.Config as cms

filteredLayerClustersHADForEgamma = cms.EDProducer("FilteredLayerClustersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    LayerClustersInputMask = cms.InputTag("ticlTrackstersEMForEgamma"),
    algo_number = cms.int32(8),
    clusterFilter = cms.string('ClusterFilterByAlgoAndSize'),
    iteration_label = cms.string('HAD'),
    max_cluster_size = cms.int32(9999),
    max_layerId = cms.int32(9999),
    mightGet = cms.optional.untracked.vstring,
    min_cluster_size = cms.int32(3),
    min_layerId = cms.int32(0)
)
