import FWCore.ParameterSet.Config as cms

hltFilteredLayerClustersPassthrough = cms.EDProducer("FilteredLayerClustersProducer",
    LayerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    LayerClustersInputMask = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    algo_number = cms.vint32(6, 7, 8),
    clusterFilter = cms.string('ClusterFilterBySize'),
    iteration_label = cms.string('Passthrough'),
    max_cluster_size = cms.int32(9999),
    max_layerId = cms.int32(9999),
    mightGet = cms.optional.untracked.vstring,
    min_cluster_size = cms.int32(2),
    min_layerId = cms.int32(0)
)
