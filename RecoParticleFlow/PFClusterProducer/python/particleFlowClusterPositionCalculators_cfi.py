import FWCore.ParameterSet.Config as cms

positionCalcECAL_all_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(0.08), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
    )

positionCalcECAL_3x3_nodepth = positionCalcECAL_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
    )

positionCalcHCAL_cross_nodepth = positionCalcECAL_all_nodepth.clone(
    posCalcNCrystals = cms.int32(5),
    logWeightDenominator = cms.double(0.8)
    )

positionCalcHCAL_all_nodepth = positionCalcHCAL_cross_nodepth.clone(
    posCalcNCrystals = cms.int32(-1)
    )

positionCalcHO_cross_nodepth = positionCalcECAL_all_nodepth.clone(
    posCalcNCrystals = cms.int32(5),
    logWeightDenominator = cms.double(0.5)
    )

positionCalcHO_all_nodepth = positionCalcHO_cross_nodepth.clone(
    posCalcNCrystals = cms.int32(-1)
    )

positionCalcHF_cross_nodepth = positionCalcECAL_all_nodepth.clone(
    posCalcNCrystals = cms.int32(5),
    logWeightDenominator = cms.double(0.8)
    )

positionCalcHF_all_nodepth = positionCalcHF_cross_nodepth.clone(
    posCalcNCrystals = cms.int32(-1)
    )

positionCalcPS_all_nodepth = positionCalcECAL_all_nodepth.clone(
    logWeightDenominator = cms.double(6e-5)
    )

positionCalcECAL_all_withdepth = cms.PSet(
    algoName = cms.string("ECAL2DPositionCalcWithDepthCorr"),
    ##
    minFractionInCalc = cms.double(0.0),
    minAllowedNormalization = cms.double(0.0),
    T0_EB = cms.double(7.4),
    T0_EE = cms.double(3.1),
    T0_ES = cms.double(1.2),
    W0 = cms.double(4.2),
    X0 = cms.double(0.89)
    )
