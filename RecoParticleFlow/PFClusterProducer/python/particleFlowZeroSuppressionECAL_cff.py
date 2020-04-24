import FWCore.ParameterSet.Config as cms

pfZeroSuppressionThresholds_EB = [0.080]*170
pfZeroSuppressionThresholds_EEminus = [0.300]*39
pfZeroSuppressionThresholds_EEplus = pfZeroSuppressionThresholds_EEminus



_pfZeroSuppressionThresholds_EB_2017 = [0.250]*170
_pfZeroSuppressionThresholds_EEminus_2017 = [1.25023,   1.25033,   1.25047,   1.25068,   1.25097,   1.25139,   1.25199,   1.25286,   1.2541,   1.25587,  # rings 170-179 (EE-) / 209-218 (EE+) 
                                             1.25842,   1.26207,   1.26729,   1.27479,   1.28553,   1.30092,   1.32299,   1.35462,   1.39995,  1.46493,  # rings 180-189 (EE-) / 219-228 (EE+)
                                             1.55807,   1.69156,   1.88291,   2.15716,   2.55027,   3.11371,   3.92131,   5.07887,   6.73803,  9.11615,  # rings 190-199 (EE-) / 229-238 (EE+)
                                             10,   10,   10,   10,   10,   10,   10,   10,   10 ]  # rings 200-208 (EE-) / 239-247 (EE+)
_pfZeroSuppressionThresholds_EEplus_2017 = _pfZeroSuppressionThresholds_EEminus_2017



_pfZeroSuppressionThresholds_EB_2018_B = [0.140]*170
_pfZeroSuppressionThresholds_EEminus_2018_B = [0.11, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.17, 0.18, 0.18, 0.19, 0.19, 0.20, 0.22, 0.23, 0.25, 0.27, 0.29, 0.31, 0.34, 0.36, 0.39, 0.42, 0.45, 0.50, 0.57, 0.68, 0.84, 1.07, 1.40, 1.88, 2.55, 3.47, 4.73, 6.42, 8.65, 11.6, 15.4]  
_pfZeroSuppressionThresholds_EEplus_2018_B = _pfZeroSuppressionThresholds_EEminus_2018_B



particle_flow_zero_suppression_ECAL = cms.PSet(
    thresholds = cms.vdouble(pfZeroSuppressionThresholds_EB + pfZeroSuppressionThresholds_EEminus + pfZeroSuppressionThresholds_EEplus
        )
    )

_particle_flow_zero_suppression_ECAL_2017 = cms.PSet(
    thresholds = cms.vdouble(_pfZeroSuppressionThresholds_EB_2017 + _pfZeroSuppressionThresholds_EEminus_2017 + _pfZeroSuppressionThresholds_EEplus_2017
        )
    )

_particle_flow_zero_suppression_ECAL_2018_B = cms.PSet(
    thresholds = cms.vdouble(_pfZeroSuppressionThresholds_EB_2018_B + _pfZeroSuppressionThresholds_EEminus_2018_B + _pfZeroSuppressionThresholds_EEplus_2018_B
        )
    )

from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
run2_ECAL_2017.toReplaceWith(particle_flow_zero_suppression_ECAL, _particle_flow_zero_suppression_ECAL_2017)

from Configuration.Eras.Modifier_phase2_ecal_cff import phase2_ecal
phase2_ecal.toReplaceWith(particle_flow_zero_suppression_ECAL, _particle_flow_zero_suppression_ECAL_2017)
