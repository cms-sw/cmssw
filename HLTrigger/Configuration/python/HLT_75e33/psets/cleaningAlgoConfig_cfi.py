import FWCore.ParameterSet.Config as cms

cleaningAlgoConfig = cms.PSet(
    cThreshold_barrel = cms.double(4),
    cThreshold_double = cms.double(10),
    cThreshold_endcap = cms.double(15),
    e4e1Threshold_barrel = cms.double(0.08),
    e4e1Threshold_endcap = cms.double(0.3),
    e4e1_a_barrel = cms.double(0.02),
    e4e1_a_endcap = cms.double(0.02),
    e4e1_b_barrel = cms.double(0.02),
    e4e1_b_endcap = cms.double(-0.0125),
    e6e2thresh = cms.double(0.04),
    ignoreOutOfTimeThresh = cms.double(1000000000.0),
    tightenCrack_e1_double = cms.double(2),
    tightenCrack_e1_single = cms.double(1),
    tightenCrack_e4e1_single = cms.double(2.5),
    tightenCrack_e6e2_double = cms.double(3)
)