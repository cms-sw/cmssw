import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3D = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    algo_number = 8,
    iteration_label = "CLUE3D"
)

# CA - PATTERN RECOGNITION

ticlTrackstersCLUE3D = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3D:CLUE3D",
    seeding_regions = "ticlSeedingGlobal",
    itername = "CLUE3D",
    patternRecognitionBy = "CLUE3D"
)

# MULTICLUSTERS

ticlMultiClustersFromTrackstersCLUE3D = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersCLUE3D"
)

ticlCLUE3DStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3D
    ,ticlTrackstersCLUE3D
    ,ticlMultiClustersFromTrackstersCLUE3D)

