import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingTrk, ticlSeedingTrkHFNose
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersTrk = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 3, # inclusive
    algo_number = [7, 6], # reco::CaloCluster::hgcal_em, reco::CaloCluster::hgcal_had,
    LayerClustersInputMask = 'ticlTrackstersEM',
    iteration_label = "Trk"
)

# CA - PATTERN RECOGNITION

ticlTrackstersTrk = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersTrk:Trk",
    seeding_regions = "ticlSeedingTrk",
    original_mask = 'ticlTrackstersEM',
    pluginPatternRecognitionByCA = dict(
        filter_on_categories = [2, 4], # filter muons and charged hadrons
        pid_threshold = 0.0,
        skip_layers = 3,
        min_layers_per_trackster = 10,
        min_cos_theta = 0.866, # ~30 degrees
        min_cos_pointing = 0.798, # ~ 37 degrees
        max_delta_time = -1.,
        algo_verbosity = 2,
        oneTracksterPerTrackSeed = True,
        promoteEmptyRegionToTrackster = True
    ),
    itername = "Trk"
)

ticlTrkStepTask = cms.Task(ticlSeedingTrk
    ,filteredLayerClustersTrk
    ,ticlTrackstersTrk)
    
# HFNOSE CLUSTER FILTERING/MASKING

filteredLayerClustersHFNoseTrk = filteredLayerClustersTrk.clone(
    LayerClusters = 'hgcalLayerClustersHFNose',
    LayerClustersInputMask = 'ticlTrackstersHFNoseEM',
    min_cluster_size = 2, # inclusive
    algo_number = [9], # reco::CaloCluster::hfnose
    iteration_label = "Trkn"
)

# HFNOSE CA - PATTERN RECOGNITION

ticlTrackstersHFNoseTrk = ticlTrackstersTrk.clone(
    detector = "HFNose",
    layer_clusters = "hgcalLayerClustersHFNose",
    layer_clusters_hfnose_tiles = "ticlLayerTileHFNose",
    original_mask = 'ticlTrackstersHFNoseEM',
    filtered_mask = "filteredLayerClustersHFNoseTrk:Trkn",
    seeding_regions = "ticlSeedingTrkHFNose",
    time_layerclusters = "hgcalLayerClustersHFNose:timeLayerCluster",
    pluginPatternRecognitionByCA = dict(
        filter_on_categories = [2, 4], # filter muons and charged hadrons
        pid_threshold = 0.0,
        skip_layers = 1,
        min_layers_per_trackster = 5,
        oneTracksterPerTrackSeed = True,
        promoteEmptyRegionToTrackster = True
    ),
    itername = "Trkn"
)

ticlHFNoseTrkStepTask = cms.Task(ticlSeedingTrkHFNose
    ,filteredLayerClustersHFNoseTrk
    ,ticlTrackstersHFNoseTrk)


