import FWCore.ParameterSet.Config as cms

ppsAlignmentConfigESSource = cms.ESSource("PPSAlignmentConfigESSource",
	label = cms.string('reference'),
	
	sector_45 = cms.PSet(
		rp_N = cms.PSet(
			y_cen_add = cms.double(-0.2),
			y_width_mult = cms.double(1.0),
			x_slice_min = cms.double(2.),
			x_slice_max = cms.double(16.)
		),
		rp_F = cms.PSet(
			y_cen_add = cms.double(-0.2),
			y_width_mult = cms.double(1.0),
			x_slice_min = cms.double(2.),
			x_slice_max = cms.double(16.)
		),
        cut_h_c = cms.double(0.04),
        cut_v_c = cms.double(0.07)
	),

	sector_56 = cms.PSet(
		rp_N = cms.PSet(
			y_cen_add = cms.double(-0.4),
			y_width_mult = cms.double(1.0),
			x_slice_min = cms.double(3.),
			x_slice_max = cms.double(16.5),
		),
		rp_F = cms.PSet(
			y_cen_add = cms.double(-0.4),
			y_width_mult = cms.double(1.0),
			x_slice_min = cms.double(2.5),
			x_slice_max = cms.double(16.5)
		),
        cut_h_c = cms.double(0.19),
        cut_v_c = cms.double(0.01)
	),

	x_alignment_meth_o = cms.PSet(
		rp_L_F = cms.PSet(
			x_min = cms.double(5.),
			x_max = cms.double(15.),
		),
		rp_L_N = cms.PSet(
			x_min = cms.double(5.),
			x_max = cms.double(15.),
		),
		rp_R_N = cms.PSet(
			x_min = cms.double(4.),
			x_max = cms.double(12.),
		),
		rp_R_F = cms.PSet(
			x_min = cms.double(4.),
			x_max = cms.double(12.),
		)
	),

	binning = cms.PSet(
		pixel_x_offset = cms.double(0.)
	)
)