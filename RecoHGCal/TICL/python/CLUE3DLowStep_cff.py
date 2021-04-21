import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3DLow = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    algo_number = 8,
    LayerClustersInputMask = 'ticlTrackstersCLUE3DHigh',
    iteration_label = "CLUE3DLow"
)

# CA - PATTERN RECOGNITION

ticlTrackstersCLUE3DLow = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3DLow:CLUE3DLow",
    seeding_regions = "ticlSeedingGlobal",
    itername = "CLUE3DLow",
    patternRecognitionBy = "CLUE3D",
    pluginPatternRecognitionByCLUE3D = dict (
        criticalDensity = 2.
    )
)

# MULTICLUSTERS

ticlMultiClustersFromTrackstersCLUE3DLow = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersCLUE3DLow"
)

ticlCLUE3DLowStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3DLow
    ,ticlTrackstersCLUE3DLow
    ,ticlMultiClustersFromTrackstersCLUE3DLow)

