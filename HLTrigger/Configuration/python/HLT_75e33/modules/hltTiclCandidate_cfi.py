import FWCore.ParameterSet.Config as cms

hltTiclCandidate = cms.EDProducer("TICLCandidateProducer",
    inferenceAlgo = cms.string('TracksterInferenceByPFN'),
    regressionAndPid = cms.bool(True),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
      algo_verbosity = cms.int32(0),
      onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/id_v0.onnx'),
      onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
      inputNames = cms.vstring(
        'input',
        'input_tr_features'
      ),
      output_en = cms.vstring('enreg_output'),
      output_id = cms.vstring('pid_output'),
      eid_min_cluster_energy = cms.double(2.5),
      eid_n_layers = cms.int32(50),
      eid_n_clusters = cms.int32(10),
      doPID = cms.int32(1),
      doRegression = cms.int32(1),
      type = cms.string('TracksterInferenceByPFN')
    ),
    cutTk = cms.string('1.48 < abs(eta) < 3.0 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
    detector = cms.string('HGCAL'),
    egamma_tracksterlinks_collections = cms.VInputTag("hltTiclTracksterLinks"),
    egamma_tracksters_collections = cms.VInputTag("hltTiclTracksterLinks"),
    general_tracksterlinks_collections = cms.VInputTag("hltTiclTracksterLinks"),
    general_tracksters_collections = cms.VInputTag("hltTiclTracksterLinks"),
    interpretationDescPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        cutTk = cms.string('1.48 < abs(eta) < 3.0 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
        delta_tk_ts_interface = cms.double(0.03),
        delta_tk_ts_layer1 = cms.double(0.02),
        timing_quality_threshold = cms.double(0.5),
        type = cms.string('General')
    ),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    layer_clustersTime = cms.InputTag("hltMergeLayerClusters","timeLayerCluster"),
    mightGet = cms.optional.untracked.vstring,
    muons = cms.InputTag("hltPhase2L3Muons"),
    original_masks = cms.VInputTag("hltMergeLayerClusters:InitialLayerClustersMask"),
    propagator = cms.string('PropagatorWithMaterial'),
    timingQualityThreshold = cms.double(0.5),
    timingSoA = cms.InputTag("mtdSoA"),
    tracks = cms.InputTag("hltGeneralTracks"),
    useMTDTiming = cms.bool(False),
    useTimingAverage = cms.bool(False)
)

from Configuration.ProcessModifiers.ticlv5_TrackLinkingGNN_cff import ticl_v5_TrackLinkingGNN
ticl_v5_TrackLinkingGNN.toModify(hltTiclCandidate,
    interpretationDescPSet = cms.PSet(
        onnxTrkLinkingModelFirstDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/FirstDiskPropGNN_v0.onnx'),
        onnxTrkLinkingModelInterfaceDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/InterfaceDiskPropGNN_v0.onnx'),
        inputNames = cms.vstring('x', 'edge_index', 'edge_attr'),
        output = cms.vstring('output'),
        delta_tk_ts = cms.double(0.1),
        thr_gnn = cms.double(0.5),
        type = cms.string('GNNLink')
    )
)
