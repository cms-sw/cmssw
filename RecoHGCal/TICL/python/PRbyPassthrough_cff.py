import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersPassthrough = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterBySize",
    min_cluster_size = 2, # inclusive
    iteration_label = "Passthrough",
    LayerClustersInputMask = 'ticlTrackstersCLUE3DHigh',
)

# PATTERN RECOGNITION

ticlTrackstersPassthrough = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersPassthrough:Passthrough",
    original_mask = 'ticlTrackstersCLUE3DHigh',
    seeding_regions = "ticlSeedingGlobal",
    itername = "PassThrough",
    patternRecognitionBy = "Passthrough",
    pluginPatternRecognitionByPassthrough = dict (
        algo_verbosity = 0
    )
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticlPassthroughStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersPassthrough
    ,ticlTrackstersPassthrough)
