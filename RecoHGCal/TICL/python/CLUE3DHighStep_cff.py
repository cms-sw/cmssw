import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3DHigh = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    algo_number = 8,
    iteration_label = "CLUE3DHigh"
)

# PATTERN RECOGNITION

ticlTrackstersCLUE3DHigh = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3DHigh:CLUE3DHigh",
    seeding_regions = "ticlSeedingGlobal",
    itername = "CLUE3DHigh",
    patternRecognitionBy = "CLUE3D",
    pluginPatternRecognitionByCLUE3D = dict (
        criticalEtaPhiDistance = 0.025
    )

)

ticlCLUE3DHighStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3DHigh
    ,ticlTrackstersCLUE3DHigh)

