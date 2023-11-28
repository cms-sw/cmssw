import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingTrk, ticlSeedingTrkHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersTrkEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSizeAndLayerRange",
    min_cluster_size = 3, # inclusive
    max_layerId = 30, # inclusive
    iteration_label = "TrkEM"
)

# CA - PATTERN RECOGNITION

ticlTrackstersTrkEM = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersTrkEM:TrkEM",
    seeding_regions = "ticlSeedingTrk",
    pluginPatternRecognitionByCA = dict(
        algo_verbosity = 0,
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
        max_delta_time = 3.
    ),
    itername = "TrkEM",
)

ticlTrkEMStepTask = cms.Task(ticlSeedingTrk
    ,filteredLayerClustersTrkEM
    ,ticlTrackstersTrkEM)

# HFNOSE CLUSTER FILTERING/MASKING

filteredLayerClustersHFNoseTrkEM = filteredLayerClustersTrkEM.clone(
    LayerClusters = 'hgcalLayerClustersHFNose',
    LayerClustersInputMask = "hgcalLayerClustersHFNose:InitialLayerClustersMask",
    min_cluster_size = 3, # inclusive
    algo_number = [9], # reco::CaloCluster::hfnose
    iteration_label = "TrkEMn"
)

# HFNOSE CA - PATTERN RECOGNITION

ticlTrackstersHFNoseTrkEM = ticlTrackstersTrkEM.clone(
    detector = "HFNose",
    layer_clusters = "hgcalLayerClustersHFNose",
    layer_clusters_hfnose_tiles = "ticlLayerTileHFNose",
    original_mask = "hgcalLayerClustersHFNose:InitialLayerClustersMask",
    filtered_mask = "filteredLayerClustersHFNoseTrkEM:TrkEMn",
    seeding_regions = "ticlSeedingTrkHFNose",
    time_layerclusters = "hgcalLayerClustersHFNose:timeLayerCluster",
    itername = "TrkEMn",
    pluginPatternRecognitionByCA = dict(
        filter_on_categories = [0, 1],
        min_layers_per_trackster = 5,
        pid_threshold = 0.,
        shower_start_max_layer = 5 #inclusive
    )
)

ticlHFNoseTrkEMStepTask = cms.Task(ticlSeedingTrkHFNose
    ,filteredLayerClustersHFNoseTrkEM
    ,ticlTrackstersHFNoseTrkEM)

