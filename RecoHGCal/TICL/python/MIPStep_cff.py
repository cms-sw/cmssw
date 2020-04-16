import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersMIP = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterBySize",
    algo_number = 8,
    max_cluster_size = 2, # inclusive
    iteration_label = "MIP"
)

# CA - PATTERN RECOGNITION

ticlTrackstersMIP = _trackstersProducer.clone(
    filtered_mask = cms.InputTag("filteredLayerClustersMIP", "MIP"),
    seeding_regions = "ticlSeedingGlobal",
    missing_layers = 3,
    min_clusters_per_ntuplet = 10,
    min_cos_theta = 0.99, # ~10 degrees
    min_cos_pointing = 0.5,
    out_in_dfs = False,
    itername = "MIP",
    max_delta_time = -1
)

# MULTICLUSTERS

ticlMultiClustersFromTrackstersMIP = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersMIP"
)

ticlMIPStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersMIP
    ,ticlTrackstersMIP
    ,ticlMultiClustersFromTrackstersMIP)

