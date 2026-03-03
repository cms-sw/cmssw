import FWCore.ParameterSet.Config as cms
from ..psets.hltTiclTracksterLinksPSet_cfi import hltTiclTracksterLinksPSet 

hltTiclTracksterLinks = cms.EDProducer("TracksterLinksProducer",
    detector = cms.string('HGCAL'),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    layer_clustersTime = cms.InputTag("hltMergeLayerClusters","timeLayerCluster"),
    inferenceAlgo = cms.string(''),
    linkingPSet = hltTiclTracksterLinksPSet,
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
        onnxPIDModelPath = cms.string(''),
        onnxEnergyModelPath = cms.string(''),
        inputNames  = cms.vstring('input'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        type = cms.string('TracksterInferenceByDNN')
    ),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
        algo_verbosity = cms.int32(0),
        onnxPIDModelPath = cms.string(''),
        onnxEnergyModelPath = cms.string(''),
        inputNames  = cms.vstring('input','input_tr_features'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        type = cms.string('TracksterInferenceByPFN'),
        miniBatchSize = cms.untracked.int32(64)
    ),
    mightGet = cms.optional.untracked.vstring,
    original_masks = cms.VInputTag("hltMergeLayerClusters:InitialLayerClustersMask"),
    propagator = cms.string('PropagatorWithMaterial'),
    regressionAndPid = cms.bool(False),
    tracksters_collections = cms.VInputTag("hltTiclTrackstersCLUE3DHigh", "hltTiclTrackstersRecovery")
)


