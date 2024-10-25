import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer

filteredLayerClustersCLUE3DBarrel = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgo",
    algo_number = [10, 11],
    iteration_label = "CLUE3DBarrel",
    max_layerid = 5
)

# PATTERN RECOGNITION
ticlTrackstersCLUE3DBarrel = _tracksters.clone(
    filtered_mask = "filteredLayerClustersCLUE3DBarrel:CLUE3DBarrel",
    seedins_regions = "ticlSeedingGlobal",
    itername = "Barrel",
    patternRecognitionBy = "CLUE3D",
    pluginPatternRecognitionByCLUE3D = dict(
        criticalDensity = [0.5, 0.5, 0.5],
        criticalSelfDensity = [0., 0., 0.],
        criticalEtaPhiDistance = [3 * 0.0175, 3 * 0.087, 3 * 0.087, 3 * 0.087],
        nearestHigherOnSameLayer = False,
        densityOnSameLayer = False,
        minNumLayerCluster = [1, 1, 1],
        useAbsoluteProjectiveScale = False,
        densitySiblingLayers = [4, 4, 4]
    )
)  
