import FWCore.ParameterSet.Config as cms

hltTiclTrackstersRecovery = cms.EDProducer("TrackstersProducer",
    detector = cms.string('HGCAL'),
    filtered_mask = cms.InputTag("hltFilteredLayerClustersRecovery","Recovery"),
    itername = cms.string('Recovery'),
    layer_clusters = cms.InputTag("hltHgcalMergeLayerClusters"),
    layer_clusters_hfnose_tiles = cms.InputTag("ticlLayerTileHFNose"),
    layer_clusters_tiles = cms.InputTag("hltTiclLayerTileProducer"),
    mightGet = cms.optional.untracked.vstring,
    original_mask = cms.InputTag("hltTiclTrackstersCLUE3DHigh"),
    patternRecognitionBy = cms.string('Recovery'),
    inferenceAlgo = cms.string('TracksterInferenceByDNN'),
    pluginPatternRecognitionByCA = cms.PSet(
        algo_verbosity = cms.int32(0),
        computeLocalTime = cms.bool(True),
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
        computeLocalTime = cms.bool(True),
        criticalDensity = cms.vdouble(4, 4, 4),
        criticalEtaPhiDistance = cms.vdouble(0.025, 0.025, 0.025),
        criticalSelfDensity = cms.vdouble(0.15, 0.15, 0.15),
        criticalXYDistance = cms.vdouble(1.8, 1.8, 1.8),
        criticalZDistanceLyr = cms.vint32(5, 5, 5),
        cutHadProb = cms.double(0.5),
        densityEtaPhiDistanceSqr = cms.vdouble(0.0008, 0.0008, 0.0008),
        densityOnSameLayer = cms.bool(False),
        densitySiblingLayers = cms.vint32(3, 3, 3),
        densityXYDistanceSqr = cms.vdouble(3.24, 3.24, 3.24),
        doPidCut = cms.bool(False),
        kernelDensityFactor = cms.vdouble(0.2, 0.2, 0.2),
        minNumLayerCluster = cms.vint32(2, 2, 2),
        nearestHigherOnSameLayer = cms.bool(False),
        outlierMultiplier = cms.vdouble(2, 2, 2),
        rescaleDensityByZ = cms.bool(False),
        type = cms.string('CLUE3D'),
        useAbsoluteProjectiveScale = cms.bool(True),
        useClusterDimensionXY = cms.bool(False)
    ),
    pluginPatternRecognitionByFastJet = cms.PSet(
        algo_verbosity = cms.int32(0),
        antikt_radius = cms.double(0.09),
        computeLocalTime = cms.bool(True),
        minNumLayerCluster = cms.int32(5),
        type = cms.string('FastJet')
    ),
    pluginPatternRecognitionByRecovery = cms.PSet(
        algo_verbosity = cms.int32(0),
        type = cms.string('Recovery')
    ),

    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
	onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/patternrecognition/id_v0.onnx'),
        onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/patternrecognition/energy_v0.onnx'),
        inputNames  = cms.vstring('input'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        doPID = cms.int32(0),
        doRegression = cms.int32(0),
        type = cms.string('TracksterInferenceByDNN')
    ),
    pluginInferenceAlgoTracksterInferenceByANN = cms.PSet(
      algo_verbosity = cms.int32(0),
      type = cms.string('TracksterInferenceByANN')
    
    ),
    seeding_regions = cms.InputTag("hltTiclSeedingGlobal"),
    time_layerclusters = cms.InputTag("hltHgcalMergeLayerClusters","timeLayerCluster")
)
