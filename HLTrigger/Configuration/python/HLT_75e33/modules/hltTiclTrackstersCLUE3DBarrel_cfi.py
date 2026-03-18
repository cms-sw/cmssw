import FWCore.ParameterSet.Config as cms

hltTiclTrackstersCLUE3DBarrel = cms.EDProducer("TrackstersProducer",
    detector = cms.string('Barrel'),
    filtered_mask = cms.InputTag("hltFilteredLayerClustersCLUE3DBarrel","CLUE3DBarrel"), # TODO change filter
    itername = cms.string('CLUE3DBarrel'),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    layer_clusters_hfnose_tiles = cms.InputTag("ticlLayerTileHFNose"),
    layer_clusters_tiles = cms.InputTag("hltTiclLayerTileProducer"),
    layer_clusters_barrel_tiles = cms.InputTag("hltTiclLayerTileBarrelProducer:ticlLayerTilesBarrel"),
    mightGet = cms.optional.untracked.vstring,
    original_mask = cms.InputTag("hltMergeLayerClusters","InitialLayerClustersMask"),
    patternRecognitionBy = cms.string('CLUE3D'),
    inferenceAlgo = cms.string('TracksterInferenceByCNNv4'),
    pluginPatternRecognitionByCA = cms.PSet(
        algo_verbosity = cms.int32(0),
        energy_em_over_total_threshold = cms.double(-1),
        etaLimitIncreaseWindow = cms.double(2.1),
        filter_on_categories = cms.vint32(0),
        max_delta_time = cms.double(3),
        max_longitudinal_sigmaPCA = cms.double(9999),
        max_missing_layers_in_trackster = cms.int32(9999),
        max_out_in_hops = cms.int32(10),
        min_cos_pointing = cms.double(-1),
        min_cos_theta = cms.double(0.915),
        min_layers_per_trackster = cms.int32(10),
        oneTracksterPerTrackSeed = cms.bool(False),
        out_in_dfs = cms.bool(True),
        pid_threshold = cms.double(0),
        promoteEmptyRegionToTrackster = cms.bool(False),
        root_doublet_max_distance_from_seed_squared = cms.double(9999),
        shower_start_max_layer = cms.int32(9999),
        siblings_maxRSquared = cms.vdouble(0.0006, 0.0006, 0.0006),
        skip_layers = cms.int32(0),
        type = cms.string('CA')
    ),
    pluginPatternRecognitionByCLUE3D = cms.PSet(
    algo_verbosity = cms.int32(0),
    criticalDensity = cms.vdouble(
      0.5,
      0.5,
      0.5
    ),
    criticalSelfDensity = cms.vdouble(
      0.,
      0.,
      0.
    ),
    densitySiblingLayers = cms.vint32(
      2,
      4,
      4
    ),
    densityEtaPhiDistanceSqr = cms.vdouble(
      0.0008,
      0.0008,
      0.0008
    ),
    densityXYDistanceSqr = cms.vdouble(
      3.24,
      3.24,
      3.24
    ),
    kernelDensityFactor = cms.vdouble(
      0.2,
      0.2,
      0.2
    ),
    densityOnSameLayer = cms.bool(False),
    nearestHigherOnSameLayer = cms.bool(False),
    useAbsoluteProjectiveScale = cms.bool(False),
    useClusterDimensionXY = cms.bool(False),
    rescaleDensityByZ = cms.bool(False),
    criticalEtaPhiDistance = cms.vdouble(
      3 * 0.0175,
      3 * 0.087,
      3 * 0.087
    ),
    criticalXYDistance = cms.vdouble(
      1.8,
      1.8,
      1.8
    ),
    criticalZDistanceLyr = cms.vint32(
      5,
      5,
      5
    ),
    outlierMultiplier = cms.vdouble(
      2,
      2,
      2
    ),
    minNumLayerCluster = cms.vint32(
      1,
      1,
      1
    ),
    computeLocalTime = cms.bool(False),
    doPidCut = cms.bool(False),
    cutHadProb = cms.double(999.),
    type = cms.string('CLUE3D')

    ),
    pluginPatternRecognitionByFastJet = cms.PSet(
        algo_verbosity = cms.int32(0),
        antikt_radius = cms.double(0.09),
        minNumLayerCluster = cms.int32(0),
        type = cms.string('FastJet')
    ),
    pluginInferenceAlgoTracksterInferenceByCNNv4 = cms.PSet(
        algo_verbosity = cms.int32(0),
        onnxModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv4/onnx_models/energy_id_v0.onnx'),
        inputNames  = cms.vstring('input:0'),
        outputNames = cms.vstring("output/regressed_energy:0", "output/id_probabilities:0"),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        doPID = cms.int32(1),
        doRegression = cms.int32(0),
        type = cms.string('TracksterInferenceByCNNv4')
    ),
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
	onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/patternrecognition/id_v0.onnx'),
        onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/patternrecognition/energy_v0.onnx'),
        inputNames  = cms.vstring('input'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        doPID = cms.int32(0),
        doRegression = cms.int32(0),
        type = cms.string('TracksterInferenceByDNN')
    ),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
        algo_verbosity = cms.int32(0),
        onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/id_v0.onnx'),
        onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/energy_v0.onnx'),
        inputNames  = cms.vstring('input','input_tr_features'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        doPID = cms.int32(0),
        doRegression = cms.int32(0),
        type = cms.string('TracksterInferenceByPFN')
    ),
    pluginInferenceAlgoTracksterInferenceByANN = cms.PSet(
        algo_verbosity = cms.int32(0),
        type = cms.string('TracksterInferenceByANN')

    ),
    seeding_regions = cms.InputTag("hltTiclSeedingGlobal"),
    time_layerclusters = cms.InputTag("hltMergeLayerClusters","timeLayerCluster")
    )
    
