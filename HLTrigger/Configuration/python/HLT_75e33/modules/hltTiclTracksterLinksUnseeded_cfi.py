import FWCore.ParameterSet.Config as cms

hltTiclTracksterLinksUnseeded = cms.EDProducer("TracksterLinksProducer",
    detector = cms.string('HGCAL'),
    layer_clusters = cms.InputTag("hltHgcalMergeLayerClusters"),
    layer_clustersTime = cms.InputTag("hltHgcalMergeLayerClusters","timeLayerCluster"),
    inferenceAlgo = cms.string('TracksterInferenceByDNN'),
    linkingPSet = cms.PSet(
      cylinder_radius_sqr_split = cms.double(9),
      proj_distance_split = cms.double(5),
      track_time_quality_threshold = cms.double(0.5),
      wind = cms.double(0.072),
      min_num_lcs = cms.uint32(15),
      min_trackster_energy = cms.double(20),
      pca_quality_th = cms.double(0.85),
      dot_prod_th = cms.double(0.97),
      deltaRxy = cms.double(4),
      lower_boundary = cms.vdouble(
        20,
        10
      ),
      upper_boundary = cms.vdouble(
        150,
        100
      ),
      upper_distance_projective_sqr = cms.vdouble(
        30,
        60
      ),
      lower_distance_projective_sqr = cms.vdouble(
        30,
        30
      ),
      min_distance_z = cms.vdouble(
        35,
        35
      ),
      upper_distance_projective_sqr_closest_points = cms.vdouble(
        10,
        30
      ),
      lower_distance_projective_sqr_closest_points = cms.vdouble(
        10,
        50
      ),
      max_z_distance_closest_points = cms.vdouble(
        35,
        35
      ),
      cylinder_radius_sqr = cms.vdouble(
        9,
        15
      ),
      algo_verbosity = cms.int32(0),
      type = cms.string('Skeletons')
    ),
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
    original_masks = cms.VInputTag("hltHgcalMergeLayerClusters:InitialLayerClustersMask"),
    propagator = cms.string('PropagatorWithMaterial'),
    regressionAndPid = cms.bool(True),
    tracksters_collections = cms.VInputTag("hltTiclTrackstersCLUE3DHigh", "hltTiclTrackstersPassthrough")
)


