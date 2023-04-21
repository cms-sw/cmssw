import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSizeAndLayerRange",
    min_cluster_size = 3, # inclusive
    max_layerId = 30, # inclusive
    algo_number = [7, 6], # reco::CaloCluster::hgcal_em, reco::CaloCluster::hgcal_had,
    LayerClustersInputMask = 'ticlTrackstersTrkEM',
    iteration_label = "EM"
)

# CA - PATTERN RECOGNITION

ticlTrackstersEM = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersEM:EM",
    original_mask = 'ticlTrackstersTrkEM',
    seeding_regions = "ticlSeedingGlobal",
    pluginPatternRecognitionByCA = dict(
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
        algo_verbosity = 0
    ),
    itername = "EM"
)

ticlEMStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersEM
    ,ticlTrackstersEM)

# HFNOSE CLUSTER FILTERING/MASKING

filteredLayerClustersHFNoseEM = filteredLayerClustersEM.clone(
    LayerClusters = 'hgcalLayerClustersHFNose',
    LayerClustersInputMask = 'ticlTrackstersHFNoseTrkEM',
    min_cluster_size = 3, # inclusive
    algo_number = [9], # reco::CaloCluster::hfnose
    iteration_label = "EMn"
)

# HFNOSE CA - PATTERN RECOGNITION

ticlTrackstersHFNoseEM = ticlTrackstersEM.clone(
    detector = "HFNose",
    layer_clusters = "hgcalLayerClustersHFNose",
    layer_clusters_hfnose_tiles = "ticlLayerTileHFNose",
    original_mask = "ticlTrackstersHFNoseTrkEM",
    filtered_mask = "filteredLayerClustersHFNoseEM:EMn",
    seeding_regions = "ticlSeedingGlobalHFNose",
    time_layerclusters = "hgcalLayerClustersHFNose:timeLayerCluster",
    itername = "EMn",
    pluginPatternRecognitionByCA = dict(
       filter_on_categories = [0, 1],
       min_layers_per_trackster = 5,
       pid_threshold = 0.,
       min_cos_pointing = 0.9845, # ~10 degrees
       shower_start_max_layer = 4 ### inclusive
    )
)

ticlHFNoseEMStepTask = cms.Task(ticlSeedingGlobalHFNose
                              ,filteredLayerClustersHFNoseEM
                              ,ticlTrackstersHFNoseEM
)
