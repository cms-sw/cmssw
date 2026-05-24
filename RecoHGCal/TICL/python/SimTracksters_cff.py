import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.simTrackstersProducer_cfi import simTrackstersProducer as _simTrackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer


filteredLayerClustersSimTracksters = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 0, # inclusive
    iteration_label = "ticlSimTracksters"
)

ticlSimTracksters = _simTrackstersProducer.clone(
    computeLocalTime = cms.bool(True),
    simClusterCollections =  cms.VPSet(
      cms.PSet( # creates a SimTrackster for every SimCluster. Creates a lot of SImTrackster (including backscattering etc)
        outputProductLabel = cms.string('fromLegacySimCluster'),
        tracksterIterationIndex = cms.int32(5), #  See Trackster.h (ticl::Trackster::IterationIndex enum). 5=SIM (ie from SimCluster), 6=SIM_CP (ie from CaloParticle)
        simClusterCollection = cms.InputTag('mix', 'MergedCaloTruth'),
        simTracksterBoundaryTime = cms.string("boundaryTime"),
        simClusterToLayerClusterAssociationMap = cms.InputTag('layerClusterSimClusterAssociationProducer'),
      ),
      cms.PSet( # the "fromBoundarySimCluster" output is what resembles the most the old behaviour : if CaloParticle simTrack crosses boundary, makes only one trackster (ignore any backscattering); 
       # else one trackster per simtrack crossing boundary (collapsing tracks that leave&reenter, different than previous behaviour that created a SimTrackster for backscattering that re-entered the calorimeter volume)
        outputProductLabel = cms.string('fromBoundarySimCluster'),
        tracksterIterationIndex = cms.int32(5),
        simClusterCollection = cms.InputTag('mix', 'MergedCaloTruthBoundaryTrackSimCluster'),
        simTracksterBoundaryTime = cms.string("boundaryTime"),
        simClusterToLayerClusterAssociationMap = cms.InputTag('layerClusterBoundaryTrackSimClusterAssociationProducer')
      ),
      cms.PSet(
        outputProductLabel = cms.string('fromMergedSimCluster'),
        tracksterIterationIndex = cms.int32(5),
        simClusterCollection = cms.InputTag('mix', 'MergedCaloTruthMergedSimCluster'),
        simTracksterBoundaryTime = cms.string("boundaryTime"), # will pick the time of the first simtrack
        simClusterToLayerClusterAssociationMap = cms.InputTag('layerClusterMergedSimClusterAssociationProducer')
      ),
      cms.PSet(
        outputProductLabel = cms.string('fromCaloParticle'),
        tracksterIterationIndex = cms.int32(6),
        simClusterCollection = cms.InputTag('mix', 'MergedCaloTruthCaloParticle'),
        simTracksterBoundaryTime = cms.string("simVertexTime"),
        simClusterToLayerClusterAssociationMap = cms.InputTag('layerClusterCaloParticleSimClusterAssociationProducer')
      ),
    ),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(ticlSimTracksters, simClusterCollections={
  i : dict(simClusterCollection=cms.InputTag("mixData", pset.simClusterCollection.productInstanceLabel, pset.simClusterCollection.processName))
  for i, pset in enumerate(ticlSimTracksters.simClusterCollections)
})
premix_stage2.toModify(ticlSimTracksters, caloParticles = "mixData:MergedCaloTruth")

_simTrackstersCollections = [cms.InputTag("ticlSimTracksters", pset.outputProductLabel) for pset in ticlSimTracksters.simClusterCollections]

from RecoHGCal.TICL.simTICLCandidateProducerUsingSimCluster_cfi import simTICLCandidateProducerUsingSimCluster as _simTICLCandidateProducerUsingSimCluster

ticlSimTICLCandidatesFromLegacy = _simTICLCandidateProducerUsingSimCluster.clone(
    baseCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthCaloParticle'), # CaloParticle as SimCluster
    subCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruth'),  # legacy SimCluster collection
    subToBaseMap = cms.InputTag('mix', 'MergedCaloTruth'),     # map SimCluster -> CaloParticle
    baseSimTracksters = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    baseSimTracksterToBaseSimObject_map = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    baseSimTracksterToCaloParticle_map = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    subSimTracksters = cms.InputTag('ticlSimTracksters', 'fromLegacySimCluster'),
    subSimTracksterToSubSimObject_map = cms.InputTag('ticlSimTracksters', 'fromLegacySimCluster'),
)
ticlSimTICLCandidatesFromBoundary = _simTICLCandidateProducerUsingSimCluster.clone(
    baseCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthCaloParticle'), # CaloParticle as SimCluster
    subCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthBoundaryTrackSimCluster'),  # legacy SimCluster collection
    subToBaseMap = cms.InputTag('mix', 'MergedCaloTruthBoundaryTrackSimCluster'),     # map SimCluster -> CaloParticle
    baseSimTracksters = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    baseSimTracksterToBaseSimObject_map = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    baseSimTracksterToCaloParticle_map = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    subSimTracksters = cms.InputTag('ticlSimTracksters', 'fromBoundarySimCluster'),
    subSimTracksterToSubSimObject_map = cms.InputTag('ticlSimTracksters', 'fromBoundarySimCluster'),
)
ticlSimTICLCandidatesFromMergedSimCluster = _simTICLCandidateProducerUsingSimCluster.clone(
    baseCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthCaloParticle'), # CaloParticle as SimCluster
    subCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthMergedSimCluster'),  # legacy SimCluster collection
    subToBaseMap = cms.InputTag('mix', 'MergedCaloTruthMergedSimCluster'),     # map SimCluster -> CaloParticle
    baseSimTracksters = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    baseSimTracksterToBaseSimObject_map = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    baseSimTracksterToCaloParticle_map = cms.InputTag('ticlSimTracksters', 'fromCaloParticle'),
    subSimTracksters = cms.InputTag('ticlSimTracksters', 'fromMergedSimCluster'),
    subSimTracksterToSubSimObject_map = cms.InputTag('ticlSimTracksters', 'fromMergedSimCluster'),
)

for _simProducer in [ticlSimTICLCandidatesFromLegacy, ticlSimTICLCandidatesFromBoundary, ticlSimTICLCandidatesFromMergedSimCluster]:
  premix_stage2.toModify(_simProducer,
    baseCaloSimObjects=cms.InputTag("mixData", _simProducer.baseCaloSimObjects.productInstanceLabel),
    subCaloSimObjects=cms.InputTag("mixData", _simProducer.subCaloSimObjects.productInstanceLabel),
    subToBaseMap=cms.InputTag("mixData", _simProducer.subToBaseMap.productInstanceLabel),
    # MtdSimTracksters=cms.InputTag("mixData", _simProducer.MtdSimTracksters.productInstanceLabel) # MTD SimTracksters do not have premixing for now
  )

ticlSimTrackstersTask = cms.Task(filteredLayerClustersSimTracksters, ticlSimTracksters, ticlSimTICLCandidatesFromLegacy, ticlSimTICLCandidatesFromBoundary, ticlSimTICLCandidatesFromMergedSimCluster)

# BARREL

filteredLayerClustersSimTrackstersBarrel = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgo",
    algo_number = [10, 11],
    iteration_label = "ticlSimTrackstersBarrel",
    max_layerId = 5
)


ticlSimTrackstersBarrel = ticlSimTracksters.clone(
    computeLocalTime = False,
    filtered_mask = "filteredLayerClustersSimTrackstersBarrel:ticlSimTrackstersBarrel",
    cutTk = cms.string('abs(eta) < 1.48 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5')
)

ticlSimTICLCandidatesFromBoundaryBarrel = ticlSimTICLCandidatesFromBoundary.clone(
    baseSimTracksters = cms.InputTag('ticlSimTrackstersBarrel', 'fromCaloParticle'),
    baseSimTracksterToBaseSimObject_map = cms.InputTag('ticlSimTrackstersBarrel', 'fromCaloParticle'),
    baseSimTracksterToCaloParticle_map = cms.InputTag('ticlSimTrackstersBarrel', 'fromCaloParticle'),
    subSimTracksters = cms.InputTag('ticlSimTrackstersBarrel', 'fromBoundarySimCluster'),
    subSimTracksterToSubSimObject_map = cms.InputTag('ticlSimTrackstersBarrel', 'fromBoundarySimCluster'),
)


from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
_ticlSimTrackstersTask = ticlSimTrackstersTask.copy()
_ticlSimTrackstersTask.add(filteredLayerClustersSimTrackstersBarrel, ticlSimTrackstersBarrel, ticlSimTICLCandidatesFromBoundaryBarrel)
ticl_barrel.toReplaceWith(ticlSimTrackstersTask, _ticlSimTrackstersTask)
     
