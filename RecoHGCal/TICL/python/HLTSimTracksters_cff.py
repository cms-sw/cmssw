import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.simTrackstersProducer_cfi import simTrackstersProducer as _simTrackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from Validation.RecoTrack.associators_cff import hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation
from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import simHitTPAssocProducer

# CA - PATTERN RECOGNITION

hltFilteredLayerClustersSimTracksters = _filteredLayerClustersProducer.clone(
    LayerClusters = cms.InputTag("hltMergeLayerClusters"),
    LayerClustersInputMask = cms.InputTag("hltMergeLayerClusters","InitialLayerClustersMask"),
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 0, # inclusive
    iteration_label = "hltTiclSimTracksters"
)

tpToHltGeneralTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltGeneralTracks"
)

hltTiclSimTracksters = _simTrackstersProducer.clone(
    filtered_mask = cms.InputTag("hltFilteredLayerClustersSimTracksters","hltTiclSimTracksters"),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    time_layerclusters = cms.InputTag("hltMergeLayerClusters","timeLayerCluster"),
    simTrackToTPMap = cms.InputTag("simHitTPAssocProducer","simTrackToTP"),
    recoTracks = cms.InputTag("hltGeneralTracks"),
    tpToTrack = cms.InputTag("tpToHltGeneralTrackAssociation"),
    computeLocalTime = cms.bool(True),
    simClusterCollections =  cms.VPSet(
      # cms.PSet( # associator only for backwards compatibility (to emulate old behaviour where sometimes SimCluster had a SimTrack without crossedBoundary flag)
      #   outputProductLabel = cms.string('fromLegacySimCluster'),
      #   tracksterIterationIndex = cms.int32(5), #  See Trackster.h (ticl::Trackster::IterationIndex enum). 5=SIM (ie from SimCluster), 6=SIM_CP (ie from CaloParticle)
      #   simClusterCollection = cms.InputTag('mix', 'MergedCaloTruth'),
      #   simTracksterBoundaryTime = cms.string("boundaryTime"),
      #   simClusterToLayerClusterAssociationMap = cms.InputTag('hltHGCalLayerClusterLegacySimClusterAssociation'),
      # ),
      cms.PSet(
        outputProductLabel = cms.string('fromBoundarySimCluster'),
        tracksterIterationIndex = cms.int32(5),
        simClusterCollection = cms.InputTag('mix', 'MergedCaloTruthBoundaryTrackSimCluster'),
        simTracksterBoundaryTime = cms.string("boundaryTime"),
        simClusterToLayerClusterAssociationMap = cms.InputTag('hltHGCalLayerClusterSimClusterAssociation')
      ),
      cms.PSet(
        outputProductLabel = cms.string('fromCaloParticle'),
        tracksterIterationIndex = cms.int32(6),
        simClusterCollection = cms.InputTag('mix', 'MergedCaloTruthCaloParticle'),
        simTracksterBoundaryTime = cms.string("simVertexTime"),
        simClusterToLayerClusterAssociationMap = cms.InputTag('hltHGCalLayerClusterCaloParticleAssociation')
      ),
    ),
)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hltTiclSimTracksters, simClusterCollections={
  i : dict(simClusterCollection=cms.InputTag("mixData", pset.simClusterCollection.productInstanceLabel, pset.simClusterCollection.processName))
  for i, pset in enumerate(hltTiclSimTracksters.simClusterCollections)
})


from RecoHGCal.TICL.simTICLCandidateProducerUsingSimCluster_cfi import simTICLCandidateProducerUsingSimCluster as _simTICLCandidateProducerUsingSimCluster

_common_hltTiclSimTICLCandidates = dict(
  baseCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthCaloParticle'), # CaloParticle as SimCluster
  baseSimTracksters = cms.InputTag('hltTiclSimTracksters', 'fromCaloParticle'),
  baseSimTracksterToBaseSimObject_map = cms.InputTag('hltTiclSimTracksters', 'fromCaloParticle'),
  baseSimTracksterToCaloParticle_map = cms.InputTag('hltTiclSimTracksters', 'fromCaloParticle'),

  MtdSimTracksters = cms.InputTag("mix", "MergedMtdTruthST"),
  recoTracks = cms.InputTag("hltGeneralTracks"), 
)
# hltTiclSimTICLCandidatesFromLegacy = _simTICLCandidateProducerUsingSimCluster.clone(
#     **_common_hltTiclSimTICLCandidates,
#     subCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruth'),  # legacy SimCluster collection
#     subToBaseMap = cms.InputTag('mix', 'MergedCaloTruth'),     # map SimCluster -> CaloParticle
#     subSimTracksters = cms.InputTag('hltTiclSimTracksters', 'fromLegacySimCluster'),
#     subSimTracksterToSubSimObject_map = cms.InputTag('hltTiclSimTracksters', 'fromLegacySimCluster'),
# )
hltTiclSimTICLCandidatesFromBoundary = _simTICLCandidateProducerUsingSimCluster.clone(
  **_common_hltTiclSimTICLCandidates,
    subCaloSimObjects = cms.InputTag('mix', 'MergedCaloTruthBoundaryTrackSimCluster'),  # legacy SimCluster collection
    subToBaseMap = cms.InputTag('mix', 'MergedCaloTruthBoundaryTrackSimCluster'),     # map SimCluster -> CaloParticle
    subSimTracksters = cms.InputTag('hltTiclSimTracksters', 'fromBoundarySimCluster'),
    subSimTracksterToSubSimObject_map = cms.InputTag('hltTiclSimTracksters', 'fromBoundarySimCluster'),
)
for _simProducer in [hltTiclSimTICLCandidatesFromBoundary]: # hltTiclSimTICLCandidatesFromLegacy
  premix_stage2.toModify(_simProducer,
    baseCaloSimObjects=cms.InputTag("mixData", _simProducer.baseCaloSimObjects.productInstanceLabel),
    subCaloSimObjects=cms.InputTag("mixData", _simProducer.subCaloSimObjects.productInstanceLabel),
    subToBaseMap=cms.InputTag("mixData", _simProducer.subToBaseMap.productInstanceLabel),
    # MtdSimTracksters=cms.InputTag("mixData", _simProducer.MtdSimTracksters.productInstanceLabel) # MTD does not yet have premixing for MtdSimTracksters
  )

from Validation.Configuration.hltHGCalSimValid_cff import *

hltTiclSimTrackstersTask = cms.Task(hltTrackAssociatorByHits,
                                    tpToHltGeneralTrackAssociation,
                                    simHitTPAssocProducer,
                                    hltHgcalAssociatorsTask,
                                    hltFilteredLayerClustersSimTracksters,
                                    hltTiclSimTracksters,
                                    hltTiclSimTICLCandidatesFromBoundary)

hltTiclSimTrackstersSeq = cms.Sequence(
    hltTiclSimTrackstersTask
)
