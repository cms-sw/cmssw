import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersRecovery = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterBySize",
    min_cluster_size = 2, # inclusive
    iteration_label = "Recovery",
    LayerClustersInputMask = 'ticlTrackstersCLUE3DHigh',
)

# PATTERN RECOGNITION

ticlTrackstersRecovery = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersRecovery:Recovery",
    original_mask = 'ticlTrackstersCLUE3DHigh',
    seeding_regions = "ticlSeedingGlobal",
    itername = "Recovery",
    patternRecognitionBy = "Recovery",
    pluginPatternRecognitionByRecovery = dict (
        algo_verbosity = 0
    )
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticlRecoveryStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersRecovery
    ,ticlTrackstersRecovery)
