import FWCore.ParameterSet.Config as cms

ppsAlignmentConfigESSource = cms.ESSource("PPSAlignmentConfigESSource",
    sequence = cms.vstring(
		"x_alignment",
		"x_alignment_relative",
		"y_alignment"
	),

	sector_45 = cms.PSet(
        rp_N = cms.PSet(
            x_max_fit_mode = cms.double(7.0),
            y_max_fit_mode = cms.double(7.0),
            slope = cms.double(0.18)
        ),
        rp_F = cms.PSet(
            x_max_fit_mode = cms.double(7.5),
            y_max_fit_mode = cms.double(7.5),
            slope = cms.double(0.17)
        ),
        slope = cms.double(0.008),

        cut_h_c = cms.double(-38.55 + 0.57 - 0.08),
        cut_v_c = cms.double(1.63 - 2.15 + 0.25)
    ),

    sector_56 = cms.PSet(
        rp_N = cms.PSet(
            x_max_fit_mode = cms.double(7.4),
            y_max_fit_mode = cms.double(7.4),
            slope = cms.double(0.34)
        ),
        rp_F = cms.PSet(
            x_max_fit_mode = cms.double(8.0),
            y_max_fit_mode = cms.double(8.0),
            slope = cms.double(0.34)
        ),
        slope = cms.double(-0.012),

        cut_h_c = cms.double(-39.26 + 0.33),
        cut_v_c = cms.double(1.49 - 1.80)
    )
)