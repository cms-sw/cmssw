import FWCore.ParameterSet.Config as cms

ticlTrackstersCLUE3DHigh = cms.EDProducer("TrackstersProducer",
    detector = cms.string('HGCAL'),
    filtered_mask = cms.InputTag("filteredLayerClustersCLUE3DHigh","CLUE3DHigh"),
    itername = cms.string('CLUE3DHigh'),
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    layer_clusters_hfnose_tiles = cms.InputTag("ticlLayerTileHFNose"),
    layer_clusters_tiles = cms.InputTag("ticlLayerTileProducer"),
    mightGet = cms.optional.untracked.vstring,
    original_mask = cms.InputTag("hgcalMergeLayerClusters","InitialLayerClustersMask"),
    patternRecognitionBy = cms.string('CLUE3D'),
    pluginPatternRecognitionByCA = cms.PSet(
        algo_verbosity = cms.int32(0),
        eid_input_name = cms.string('input'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        eid_output_name_energy = cms.string('output/regressed_energy'),
        eid_output_name_id = cms.string('output/id_probabilities'),
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
      0.6,
      0.6,
      0.6
    ),
    criticalSelfDensity = cms.vdouble(
      0.15,
      0.15,
      0.15
    ),
    densitySiblingLayers = cms.vint32(
      3,
      3,
      3
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
    useAbsoluteProjectiveScale = cms.bool(True),
    useClusterDimensionXY = cms.bool(False),
    rescaleDensityByZ = cms.bool(False),
    criticalEtaPhiDistance = cms.vdouble(
      0.025,
      0.025,
      0.025
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
      2,
      2,
      2
    ),
    eid_input_name = cms.string('input'),
    eid_output_name_energy = cms.string('output/regressed_energy'),
    eid_output_name_id = cms.string('output/id_probabilities'),
    eid_min_cluster_energy = cms.double(1),
    eid_n_layers = cms.int32(50),
    eid_n_clusters = cms.int32(10),
    doPidCut = cms.bool(True),
    cutHadProb = cms.double(999.),
    type = cms.string('CLUE3D')
  
    ),
    pluginPatternRecognitionByFastJet = cms.PSet(
        algo_verbosity = cms.int32(0),
        antikt_radius = cms.double(0.09),
        eid_input_name = cms.string('input'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        eid_output_name_energy = cms.string('output/regressed_energy'),
        eid_output_name_id = cms.string('output/id_probabilities'),
        minNumLayerCluster = cms.int32(5),
        type = cms.string('FastJet')
    ),
    seeding_regions = cms.InputTag("ticlSeedingGlobal"),
    tfDnnLabel = cms.string('tracksterSelectionTf'),
    time_layerclusters = cms.InputTag("hgcalMergeLayerClusters","timeLayerCluster")
)
