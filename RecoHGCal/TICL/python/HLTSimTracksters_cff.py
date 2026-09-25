import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.simTrackstersProducer_cfi import simTrackstersProducer as _simTrackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from Validation.RecoTrack.associators_cff import hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation, tpToHLTgsfTrackAssociation
from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import simHitTPAssocProducer
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation
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

tpHltGsfTrackAssociation = tpToHLTgsfTrackAssociation.clone(
    label_tr = cms.InputTag("hltEgammaGsfTracksL1Seeded"),
)

hltTiclSimTracksters = _simTrackstersProducer.clone(
    layerClusterCaloParticleAssociator = cms.InputTag("hltHGCalLayerClusterCaloParticleAssociation"),
    layerClusterSimClusterAssociator = cms.InputTag("hltHGCalLayerClusterSimClusterAssociation"),
    filtered_mask = cms.InputTag("hltFilteredLayerClustersSimTracksters","hltTiclSimTracksters"),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    time_layerclusters = cms.InputTag("hltMergeLayerClusters","timeLayerCluster"),
    simTrackToTPMap = cms.InputTag("simHitTPAssocProducer","simTrackToTP"),
    recoTracks = cms.InputTag("hltGeneralTracks"),
    gsfTracks  = cms.InputTag("hltEgammaGsfTracksL1Seeded"),
    simclusters = cms.InputTag("mix","MergedCaloTruth"),
    tpToTrack = cms.InputTag("tpToHltGeneralTrackAssociation"),
    tpToGsfTrack  = cms.InputTag("tpHltGsfTrackAssociation"),
    computeLocalTime = cms.bool(True)
)

from Validation.Configuration.hltHGCalSimValid_cff import *

hltTiclSimTrackstersTask = cms.Task(hltTrackAssociatorByHits,
                                    tpToHltGeneralTrackAssociation,
                                    tpHltGsfTrackAssociation,
                                    simHitTPAssocProducer,
                                    hltHgcalAssociatorsTask,
                                    hltFilteredLayerClustersSimTracksters,
                                    hltTiclSimTracksters)

hltTiclSimTrackstersSeq = cms.Sequence(
    hltTiclSimTrackstersTask
)
