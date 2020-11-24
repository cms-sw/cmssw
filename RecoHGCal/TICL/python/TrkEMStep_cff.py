import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingTrk
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersTrkEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSizeAndLayerRange",
    min_cluster_size = 3, # inclusive
    max_layerId = 30, # inclusive
    algo_number = 8,
    iteration_label = "TrkEM"
)

# CA - PATTERN RECOGNITION

ticlTrackstersTrkEM = _trackstersProducer.clone(
    filtered_mask = cms.InputTag("filteredLayerClustersTrkEM", "TrkEM"),
    seeding_regions = "ticlSeedingTrk",
    filter_on_categories = [0, 1],
    pid_threshold = 0.5,
    energy_em_over_total_threshold = 0.9,
    max_longitudinal_sigmaPCA = 10,
    shower_start_max_layer = 5, #inclusive
    max_out_in_hops = 1,
    max_missing_layers_in_trackster = 2,
    skip_layers = 2,
    min_layers_per_trackster = 10,
    min_cos_theta = 0.97,  # ~14 degrees
    min_cos_pointing = 0.94, # ~20 degrees
    root_doublet_max_distance_from_seed_squared = 2.5e-3, # dR=0.05
    max_delta_time = 3.,
    itername = "TrkEM",
    algo_verbosity = 0,
)


# MULTICLUSTERS

ticlMultiClustersFromTrackstersTrkEM = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersTrkEM"
)

ticlTrkEMStepTask = cms.Task(ticlSeedingTrk
    ,filteredLayerClustersTrkEM
    ,ticlTrackstersTrkEM
    ,ticlMultiClustersFromTrackstersTrkEM)

