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
