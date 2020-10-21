import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    algo_number = 8,
    LayerClustersInputMask = 'ticlTrackstersTrk',
    iteration_label = "EM"
)

# CA - PATTERN RECOGNITION

ticlTrackstersEM = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersEM:EM",
    original_mask = 'ticlTrackstersTrk',
    seeding_regions = "ticlSeedingGlobal",
    filter_on_categories = [0, 1],
    pid_threshold = 0.8,
    max_out_in_hops = 4,
    missing_layers = 1,
    min_clusters_per_ntuplet = 10,
    min_cos_theta = 0.978,  # ~12 degrees
    min_cos_pointing = 0.9, # ~25 degrees
    max_delta_time = 3.,
    itername = "EM",
    algo_verbosity = 0,
)

# MULTICLUSTERS

ticlMultiClustersFromTrackstersEM = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersEM"
)

ticlEMStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersEM
    ,ticlTrackstersEM
    ,ticlMultiClustersFromTrackstersEM)

filteredLayerClustersHFNoseEM = filteredLayerClustersEM.clone(
    LayerClusters = 'hgcalLayerClustersHFNose',
    LayerClustersInputMask = "hgcalLayerClustersHFNose:InitialLayerClustersMask",
    iteration_label = "EMn",
    algo_number = 9
#no tracking mask for EM for now
)

ticlTrackstersHFNoseEM = ticlTrackstersEM.clone(
    detector = "HFNose",
    layer_clusters = "hgcalLayerClustersHFNose",
    layer_clusters_hfnose_tiles = "ticlLayerTileHFNose",
    original_mask = "hgcalLayerClustersHFNose:InitialLayerClustersMask",
    filtered_mask = "filteredLayerClustersHFNoseEM:EMn",
    seeding_regions = "ticlSeedingGlobalHFNose",
    time_layerclusters = "hgcalLayerClustersHFNose:timeLayerCluster",
    min_clusters_per_ntuplet = 6
)

ticlHFNoseEMStepTask = cms.Task(ticlSeedingGlobalHFNose
                              ,filteredLayerClustersHFNoseEM
                              ,ticlTrackstersHFNoseEM
)
