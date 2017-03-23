import FWCore.ParameterSet.Config as cms

pfZeroSuppressionThresholds_EB = [0.250]*170
pfZeroSuppressionThresholds_EEminus = [1.25023 ,   1.25033 ,   1.25047 ,   1.25068 ,   1.25097 ,   1.25139 ,   1.25199 ,   1.25286 ,   1.2541 ,   1.25587 ,   1.25842 ,   1.26207 ,   1.26729 ,   1.27479 ,   1.28553 ,   1.30092 ,   1.32299 ,   1.35462 ,   1.39995 ,   1.46493 ,   1.55807 ,   1.69156 ,   1.88291 ,   2.15716 ,   2.55027 ,   3.11371 ,   3.92131 ,   5.07887 ,   6.73803 ,   9.11615 ,   10 ,   10 ,   10 ,   10 ,   10 ,   10 ,   10 ,   10 ,   10 ]
pfZeroSuppressionThresholds_EEplus = pfZeroSuppressionThresholds_EEminus

particle_flow_zero_suppression_ECAL = cms.PSet(
    thresholds = cms.vdouble(pfZeroSuppressionThresholds_EB + pfZeroSuppressionThresholds_EEminus + pfZeroSuppressionThresholds_EEplus
        )
    )
