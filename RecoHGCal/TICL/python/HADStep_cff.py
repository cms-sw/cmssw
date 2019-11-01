import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersHAD = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    algo_number = 8,
    iteration_label = "HAD",
    LayerClustersInputMask = "trackstersTrk"
)

# CA - PATTERN RECOGNITION

trackstersHAD = _trackstersProducer.clone(
    filtered_mask = cms.InputTag("filteredLayerClustersHAD", "HAD"),
    original_mask = 'trackstersTrk',
    seeding_regions = "ticlSeedingGlobal",
    missing_layers = 2,
    min_clusters_per_ntuplet = 10,
    min_cos_theta = 0.8,
    min_cos_pointing = 0.7
    )

# MULTICLUSTERS

multiClustersFromTrackstersHAD = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "trackstersHAD"
    )

HADStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersHAD
    ,trackstersHAD
    ,multiClustersFromTrackstersHAD)

