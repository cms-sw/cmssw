import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.simTrackstersProducer_cfi import simTrackstersProducer as _simTrackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer


# CA - PATTERN RECOGNITION


filteredLayerClustersSimTracksters = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 0, # inclusive
    iteration_label = "ticlSimTracksters"
)

ticlSimTracksters = _simTrackstersProducer.clone(
    computeLocalTime = cms.bool(False)
)
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(ticlSimTracksters, computeLocalTime = cms.bool(True))

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(ticlSimTracksters,
    simclusters = "mixData:MergedCaloTruth",
    caloparticles = "mixData:MergedCaloTruth",
)

ticlSimTrackstersTask = cms.Task(filteredLayerClustersSimTracksters, ticlSimTracksters)

# BARREL

filteredLayerClustersSimTrackstersBarrel = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgo",
    algo_number = [10, 11],
    iteration_label = "ticlSimTrackstersBarrel",
    max_layerId = 5
)


ticlSimTrackstersBarrel = _simTrackstersProducer.clone(
    computeLocalTime = False,
    filtered_mask = "filteredLayerClustersSimTrackstersBarrel:ticlSimTrackstersBarrel",
    layerClusterSimClusterAssociator = 'barrelLayerClusterSimClusterAssociation',
    layerClusterCaloParticleAssociator = 'barrelLayerClusterCaloParticleAssociation',
    cutTk = cms.string('abs(eta) < 1.48 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5')
)


from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
_ticlSimTrackstersTask = ticlSimTrackstersTask.copy()
_ticlSimTrackstersTask.add(filteredLayerClustersSimTrackstersBarrel, ticlSimTrackstersBarrel)
ticl_barrel.toReplaceWith(ticlSimTrackstersTask, _ticlSimTrackstersTask)
     
