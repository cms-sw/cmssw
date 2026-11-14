import FWCore.ParameterSet.Config as cms

hltTiclTracksterLinksPSet = cms.PSet(
      onnxModelPath = cms.string(""),
      cylinder_radius_sqr_split = cms.float(9),
      proj_distance_split = cms.float(5),
      track_time_quality_threshold = cms.float(0.5),
      min_num_lcs = cms.uint32(15),
      min_trackster_energy = cms.float(20),
      pca_quality_th = cms.float(0.85),
      dot_prod_th = cms.float(0.97),
      deltaRxy = cms.float(4),
      lower_boundary = cms.vfloat(
        20,
        10
      ),
      upper_boundary = cms.vfloat(
        150,
        100
      ),
      upper_distance_projective_sqr = cms.vfloat(
        4,
        60
      ),
      lower_distance_projective_sqr = cms.vfloat(
        4,
        60
      ),
      min_distance_z = cms.vfloat(
        35,
        35
      ),
      upper_distance_projective_sqr_closest_points = cms.vfloat(
        5,
        30
      ),
      lower_distance_projective_sqr_closest_points = cms.vfloat(
        10,
        50
      ),
      max_z_distance_closest_points = cms.vfloat(
        35,
        35
      ),
      cylinder_radius_sqr = cms.vfloat(
        9,
        15
      ),
      algo_verbosity = cms.int32(0),
      type = cms.string('Skeletons')
    )
