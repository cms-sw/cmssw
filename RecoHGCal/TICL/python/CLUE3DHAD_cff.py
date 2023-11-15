import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3DHAD = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterBySize",
    min_cluster_size = 2, # inclusive
    iteration_label = "CLUE3DHAD",
    LayerClustersInputMask = 'ticlTrackstersCLUE3DEM',
)

# PATTERN RECOGNITION

ticlTrackstersCLUE3DHAD = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3DHAD:CLUE3DHAD",
    original_mask = 'ticlTrackstersCLUE3DEM',
    seeding_regions = "ticlSeedingGlobal",
    itername = "CLUE3DHAD",
    patternRecognitionBy = "CLUE3D",
    pluginPatternRecognitionByCLUE3D = dict (
        criticalDensity = [0.6, 0.6, 0.6],
        criticalEtaPhiDistance = [0.025, 0.025, 0.025],
        kernelDensityFactor = [0.2, 0.2, 0.2],
        algo_verbosity = 0
    )

)

ticlCLUE3DHADStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3DHAD
    ,ticlTrackstersCLUE3DHAD)

