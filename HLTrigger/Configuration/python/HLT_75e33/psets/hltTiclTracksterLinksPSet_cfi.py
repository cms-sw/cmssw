import FWCore.ParameterSet.Config as cms

hltTiclTracksterLinksPSet = cms.PSet(
      cylinder_radius_sqr_split = cms.double(9),
      proj_distance_split = cms.double(5),
      track_time_quality_threshold = cms.double(0.5),
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
        60
      ),
      min_distance_z = cms.vdouble(
        35,
        35
      ),
      upper_distance_projective_sqr_closest_points = cms.vdouble(
        5,
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
    )
