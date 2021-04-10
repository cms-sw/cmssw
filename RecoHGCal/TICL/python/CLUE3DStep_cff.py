import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3D = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 1, # inclusive
    algo_number = 8,
    iteration_label = "CLUE3D"
)

# CA - PATTERN RECOGNITION

ticlTrackstersCLUE3D = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3D:CLUE3D",
    seeding_regions = "ticlSeedingGlobal",
    filter_on_categories = [0, 1],
    pid_threshold = 0.5,
    energy_em_over_total_threshold = 0.9,
    max_longitudinal_sigmaPCA = 10,
    shower_start_max_layer = 5, #inclusive
    max_out_in_hops = 1,
    skip_layers = 2,
    max_missing_layers_in_trackster = 1,
    min_layers_per_trackster = 10,
    min_cos_theta = 0.97,  # ~14 degrees
    min_cos_pointing = 0.9, # ~25 degrees
    max_delta_time = 3.,
    itername = "CLUE3D",
    algo_verbosity = 0,
    patternRecognitionAlgo = "CLUE3D"
)

# MULTICLUSTERS

ticlMultiClustersFromTrackstersCLUE3D = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersCLUE3D"
)

ticlCLUE3DStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3D
    ,ticlTrackstersCLUE3D
    ,ticlMultiClustersFromTrackstersCLUE3D)

