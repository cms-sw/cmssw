import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersMIP = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterBySize",
    algo_number = [7, 6], # reco::CaloCluster::hgcal_em, reco::CaloCluster::hgcal_had,
    max_cluster_size = 2, # inclusive
    iteration_label = "MIP"
)


# CA - PATTERN RECOGNITION

ticlTrackstersMIP = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersMIP:MIP",
    seeding_regions = "ticlSeedingGlobal",
    pluginPatternRecognitionByCA = dict(
      skip_layers = 3,
      min_layers_per_trackster = 10,
      min_cos_theta = 0.99, # ~10 degrees
      min_cos_pointing = 0.5,
      out_in_dfs = False,
      max_delta_time = -1
    ),
    itername = "MIP"
)

ticlMIPStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersMIP
    ,ticlTrackstersMIP)

filteredLayerClustersHFNoseMIP = filteredLayerClustersMIP.clone(
    LayerClusters = 'hgcalLayerClustersHFNose',
    LayerClustersInputMask = "hgcalLayerClustersHFNose:InitialLayerClustersMask",
    iteration_label = "MIPn",
    algo_number = [9]
)

ticlTrackstersHFNoseMIP = ticlTrackstersMIP.clone(
    detector = "HFNose",
    layer_clusters = "hgcalLayerClustersHFNose",
    layer_clusters_hfnose_tiles = "ticlLayerTileHFNose",
    original_mask = "hgcalLayerClustersHFNose:InitialLayerClustersMask",
    filtered_mask = "filteredLayerClustersHFNoseMIP:MIPn",
    seeding_regions = "ticlSeedingGlobalHFNose",
    time_layerclusters = "hgcalLayerClustersHFNose:timeLayerCluster",
    pluginPatternRecognitionByCA = dict(min_layers_per_trackster = 6)
)

ticlHFNoseMIPStepTask = cms.Task(ticlSeedingGlobalHFNose
                              ,filteredLayerClustersHFNoseMIP
                              ,ticlTrackstersHFNoseMIP
)
