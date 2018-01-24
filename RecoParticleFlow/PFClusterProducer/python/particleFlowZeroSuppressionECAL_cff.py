import FWCore.ParameterSet.Config as cms

pfZeroSuppressionThresholds_EB = [0.080]*170
pfZeroSuppressionThresholds_EEminus = [0.300]*39
pfZeroSuppressionThresholds_EEplus = pfZeroSuppressionThresholds_EEminus

#
# The thresholds have been temporarily removed (lowered to 80 MeV in EB and 300 MeV in EE,
# then overseeded by the gathering and seeding PF cluster thresholds)
# Later, we may need to reintroduce eta dependent thresholds
# to mitigate the effect of the noise
#


particle_flow_zero_suppression_ECAL = cms.PSet(
    thresholds = cms.vdouble(pfZeroSuppressionThresholds_EB + pfZeroSuppressionThresholds_EEminus + pfZeroSuppressionThresholds_EEplus
        )
    )

