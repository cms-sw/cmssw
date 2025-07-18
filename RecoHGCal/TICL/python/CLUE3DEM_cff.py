import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3DEM = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSizeAndLayerRange",
    min_cluster_size = 2, # inclusive
    iteration_label = "CLUE3DEM",
    algo_number = [6,7],
    max_layerId = 28, # inclusive
)

# PATTERN RECOGNITION

ticlTrackstersCLUE3DEM = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3DEM:CLUE3DEM",
    seeding_regions = "ticlSeedingGlobal",
    itername = "EM",
    patternRecognitionBy = "CLUE3D",
    pluginPatternRecognitionByCLUE3D = dict (
        criticalDensity = [0.6, 0.6, 0.6],
        criticalEtaPhiDistance = [0.025, 0.025, 0.025],
        kernelDensityFactor = [0.2, 0.2, 0.2],
        algo_verbosity = 0
    )

)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(ticlTrackstersCLUE3DEM.pluginPatternRecognitionByCLUE3D, computeLocalTime = cms.bool(True))

ticlCLUE3DEMStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3DEM
    ,ticlTrackstersCLUE3DEM)

