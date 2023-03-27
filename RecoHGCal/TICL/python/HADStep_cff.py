import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersHAD = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 3, # inclusive
    algo_number = [7, 6], # reco::CaloCluster::hgcal_em, reco::CaloCluster::hgcal_had,
    iteration_label = "HAD",
    LayerClustersInputMask = "ticlTrackstersTrk"
)

# CA - PATTERN RECOGNITION

ticlTrackstersHAD = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersHAD:HAD",
    original_mask = 'ticlTrackstersTrk',
    seeding_regions = "ticlSeedingGlobal",
    # For the moment we mask everything w/o requirements since we are last
#    filter_on_categories = [5], # filter neutral hadrons
#    pid_threshold = 0.7,
    pluginPatternRecognitionByCA = dict(
       skip_layers = 1,
       min_layers_per_trackster = 12,
       min_cos_theta = 0.866,    # ~30 degrees
       min_cos_pointing = 0.819, # ~35 degrees
       max_delta_time = -1
    ),
    itername = "HAD"
)

ticlHADStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersHAD
    ,ticlTrackstersHAD)

# HFNOSE CLUSTER FILTERING/MASKING

filteredLayerClustersHFNoseHAD = filteredLayerClustersHAD.clone(
    min_cluster_size = 2, # inclusive
    algo_number = [9],
    iteration_label = "HADn",
    LayerClusters = 'hgcalLayerClustersHFNose',
    LayerClustersInputMask = "ticlTrackstersHFNoseTrk"
)

# HFNOSE CA - PATTERN RECOGNITION

ticlTrackstersHFNoseHAD = ticlTrackstersHAD.clone(
    detector = "HFNose",
    layer_clusters = "hgcalLayerClustersHFNose",
    layer_clusters_hfnose_tiles = "ticlLayerTileHFNose",
    original_mask = "ticlTrackstersHFNoseTrk",
    filtered_mask = "filteredLayerClustersHFNoseHAD:HADn",
    seeding_regions = "ticlSeedingGlobalHFNose",
    time_layerclusters = "hgcalLayerClustersHFNose:timeLayerCluster",
    pluginPatternRecognitionByCA = dict(
       pid_threshold = 0.,
       skip_layers = 1,
       min_layers_per_trackster = 5,
       min_cos_theta = 0.866,    # ~30 degrees
       min_cos_pointing = 0.866, # ~30 degrees
       max_delta_time = -1
    ),
    itername = "HADn"
    )

ticlHFNoseHADStepTask = cms.Task(ticlSeedingGlobalHFNose
                                 ,filteredLayerClustersHFNoseHAD
                                 ,ticlTrackstersHFNoseHAD)
