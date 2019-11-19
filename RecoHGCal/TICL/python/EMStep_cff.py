import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    algo_number = 8,
    LayerClustersInputMask = 'trackstersTrk',
    iteration_label = "EM"
)

# CA - PATTERN RECOGNITION

trackstersEM = _trackstersProducer.clone(
    filtered_mask = cms.InputTag("filteredLayerClustersEM", "EM"),
    original_mask = 'trackstersTrk',
    seeding_regions = "ticlSeedingGlobal",
    filter_on_categories = [0, 1],
    pid_threshold = 0.8,
    max_out_in_hops = 4,
    missing_layers = 1,
    min_clusters_per_ntuplet = 10,
    min_cos_theta = 0.978,  # ~12 degrees
    min_cos_pointing = 0.9, # ~25 degrees
    max_delta_time = 0.12
)

# MULTICLUSTERS

multiClustersFromTrackstersEM = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "trackstersEM"
)

EMStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersEM
    ,trackstersEM
    ,multiClustersFromTrackstersEM)

