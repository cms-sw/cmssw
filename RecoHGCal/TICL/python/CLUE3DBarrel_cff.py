import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer

filteredLayerClustersCLUE3DBarrel = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgo",
    algo_number = [10, 11],
    iteration_label = "CLUE3DBarrel",
    max_layerId = 5
)

# PATTERN RECOGNITION
ticlTrackstersCLUE3DBarrel = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersCLUE3DBarrel:CLUE3DBarrel",
    seeding_regions = "ticlSeedingGlobal",
    itername = "CLUE3DBarrel",
    detector = "Barrel",
    patternRecognitionBy = "CLUE3D",
    layer_clusters_barrel_tiles = "ticlLayerTileBarrel:ticlLayerTilesBarrel",
    pluginPatternRecognitionByCLUE3D = dict(
        algo_verbosity = 9999,
        criticalDensity = [0.5, 0.5, 0.5],
        criticalSelfDensity = [0., 0., 0.],
        criticalEtaPhiDistance = [3 * 0.0175, 3 * 0.087, 3 * 0.087, 3 * 0.087],
        nearestHigherOnSameLayer = False,
        densityOnSameLayer = False,
        minNumLayerCluster = [1, 1, 1],
        useAbsoluteProjectiveScale = False,
        densitySiblingLayers = [2, 4, 4]
    )
)
 
ticlCLUE3DBarrelTask = cms.Task(ticlSeedingGlobal
    ,ticlTrackstersCLUE3DBarrel
) 
