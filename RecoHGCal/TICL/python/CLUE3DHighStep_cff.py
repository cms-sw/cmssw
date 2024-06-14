import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersCLUE3DHigh = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    iteration_label = "CLUE3DHigh"
)

# PATTERN RECOGNITION

ticlTrackstersCLUE3DHigh = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3DHigh:CLUE3DHigh",
    seeding_regions = "ticlSeedingGlobal",
    itername = "CLUE3DHigh",
    patternRecognitionBy = "CLUE3D",
    pluginPatternRecognitionByCLUE3D = dict (
        criticalDensity = [0.6, 0.6, 0.6],
        criticalEtaPhiDistance = [0.025, 0.025, 0.025],
        kernelDensityFactor = [0.2, 0.2, 0.2],
        algo_verbosity = 0,
        doPidCut = True,
        cutHadProb = 999
    )

)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(ticlTrackstersCLUE3DHigh.pluginPatternRecognitionByCLUE3D, computeLocalTime = cms.bool(True))
ticl_v5.toModify(ticlTrackstersCLUE3DHigh.pluginPatternRecognitionByCLUE3D, doPidCut = cms.bool(False))
ticl_v5.toModify(ticlTrackstersCLUE3DHigh.pluginPatternRecognitionByCLUE3D, usePCACleaning = cms.bool(True))

ticlCLUE3DHighStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersCLUE3DHigh
    ,ticlTrackstersCLUE3DHigh)

