import FWCore.ParameterSet.Config as cms

pfZeroSuppressionThresholds_EB = [0.080]*170
pfZeroSuppressionThresholds_EEminus = [0.300]*39
pfZeroSuppressionThresholds_EEplus = pfZeroSuppressionThresholds_EEminus

#
# These are expected to be adjusted soon, while the thresholds for older setups will remain unchanged.
#
_pfZeroSuppressionThresholds_EB_2017 = pfZeroSuppressionThresholds_EB
_pfZeroSuppressionThresholds_EEminus_2017 = pfZeroSuppressionThresholds_EEminus
_pfZeroSuppressionThresholds_EEplus_2017 = _pfZeroSuppressionThresholds_EEminus_2017


# 
# The three different set of thresholds will be used to study
# possible new thresholds of pfrechits and effects on high level objects
# The values proposed (A, B, C) are driven by expected noise levels
# A ~ 2.0 sigma noise equivalent thresholds
# B ~ 1.0 sigma noise equivalent thresholds
# C ~ 0.5 sigma noise equivalent thresholds
#


# A
_pfZeroSuppressionThresholds_EB_2018_A = [0.180]*170
_pfZeroSuppressionThresholds_EEminus_2018_A = [0.22, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.34, 0.36, 0.36, 0.38, 0.38, 0.4, 0.44, 0.46, 0.5, 0.54, 0.58, 0.62, 0.68, 0.72, 0.78, 0.84, 0.9, 1.0, 1.14, 1.36, 1.68, 2.14, 2.8, 3.76, 5.1, 6.94, 9.46, 12.84, 17.3, 23.2, 30.8]
_pfZeroSuppressionThresholds_EEplus_2018_A = _pfZeroSuppressionThresholds_EEminus_2018_A

_particle_flow_zero_suppression_ECAL_2018_A = cms.PSet(
    thresholds = cms.vdouble(_pfZeroSuppressionThresholds_EB_2018_A + _pfZeroSuppressionThresholds_EEminus_2018_A + _pfZeroSuppressionThresholds_EEplus_2018_A
        )
    )

# B
_pfZeroSuppressionThresholds_EB_2018_B = [0.140]*170
_pfZeroSuppressionThresholds_EEminus_2018_B = [0.11, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.17, 0.18, 0.18, 0.19, 0.19, 0.20, 0.22, 0.23, 0.25, 0.27, 0.29, 0.31, 0.34, 0.36, 0.39, 0.42, 0.45, 0.50, 0.57, 0.68, 0.84, 1.07, 1.40, 1.88, 2.55, 3.47, 4.73, 6.42, 8.65, 11.6, 15.4]
_pfZeroSuppressionThresholds_EEplus_2018_B = _pfZeroSuppressionThresholds_EEminus_2018_B


_particle_flow_zero_suppression_ECAL_2018_B = cms.PSet(
    thresholds = cms.vdouble(_pfZeroSuppressionThresholds_EB_2018_B + _pfZeroSuppressionThresholds_EEminus_2018_B + _pfZeroSuppressionThresholds_EEplus_2018_B
        )
    )

# C
_pfZeroSuppressionThresholds_EB_2018_C = [0.100]*170
_pfZeroSuppressionThresholds_EEminus_2018_C = [0.055, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.085, 0.09, 0.09, 0.095, 0.095, 0.1, 0.11, 0.115, 0.125, 0.135, 0.145, 0.155, 0.17, 0.18, 0.195, 0.21, 0.225, 0.25, 0.285, 0.34, 0.42, 0.535, 0.7, 0.94, 1.275, 1.735, 2.365, 3.21, 4.325, 5.8, 7.7 ]
_pfZeroSuppressionThresholds_EEplus_2018_C = _pfZeroSuppressionThresholds_EEminus_2018_C


_particle_flow_zero_suppression_ECAL_2018_C = cms.PSet(
    thresholds = cms.vdouble(_pfZeroSuppressionThresholds_EB_2018_C + _pfZeroSuppressionThresholds_EEminus_2018_C + _pfZeroSuppressionThresholds_EEplus_2018_C
        )
    )



particle_flow_zero_suppression_ECAL = cms.PSet(
    thresholds = cms.vdouble(pfZeroSuppressionThresholds_EB + pfZeroSuppressionThresholds_EEminus + pfZeroSuppressionThresholds_EEplus
        )
    )

_particle_flow_zero_suppression_ECAL_2017 = cms.PSet(
    thresholds = cms.vdouble(_pfZeroSuppressionThresholds_EB_2017 + _pfZeroSuppressionThresholds_EEminus_2017 + _pfZeroSuppressionThresholds_EEplus_2017
        )
    )

#
# The thresholds have been temporarily removed (lowered to 80 MeV in EB and 300 MeV in EE,
# then overseeded by the gathering and seeding PF cluster thresholds)
# Later, we may need to reintroduce eta dependent thresholds
# to mitigate the effect of the noise
#

from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
run2_ECAL_2017.toReplaceWith(particle_flow_zero_suppression_ECAL, _particle_flow_zero_suppression_ECAL_2017)

from Configuration.Eras.Modifier_phase2_ecal_cff import phase2_ecal
phase2_ecal.toReplaceWith(particle_flow_zero_suppression_ECAL, _particle_flow_zero_suppression_ECAL_2017)
