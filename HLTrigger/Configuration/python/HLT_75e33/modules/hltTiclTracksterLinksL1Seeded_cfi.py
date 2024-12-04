import FWCore.ParameterSet.Config as cms
from ..psets.hltTiclTracksterLinksPSet_cfi import hltTiclTracksterLinksPSet 

hltTiclTracksterLinksL1Seeded = cms.EDProducer("TracksterLinksProducer",
    detector = cms.string('HGCAL'),
    layer_clusters = cms.InputTag("hltHgcalMergeLayerClustersL1Seeded"),
    layer_clustersTime = cms.InputTag("hltHgcalMergeLayerClustersL1Seeded","timeLayerCluster"),
    inferenceAlgo = cms.string('TracksterInferenceByDNN'),
    linkingPSet = hltTiclTracksterLinksPSet,
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
        onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/linking/id_v0.onnx'),
        onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/linking/energy_v0.onnx'),
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
    mightGet = cms.optional.untracked.vstring,
    original_masks = cms.VInputTag("hltHgcalMergeLayerClustersL1Seeded:InitialLayerClustersMask"),
    propagator = cms.string('PropagatorWithMaterial'),
    regressionAndPid = cms.bool(True),
    tracksters_collections = cms.VInputTag("hltTiclTrackstersCLUE3DHighL1Seeded", "hltTiclTrackstersRecoveryL1Seeded")
)


