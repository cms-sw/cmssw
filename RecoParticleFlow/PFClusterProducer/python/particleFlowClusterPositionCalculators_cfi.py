import FWCore.ParameterSet.Config as cms

positionCalcEB_all_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(0.08), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
    )

positionCalcEE_all_nodepth = positionCalcEB_all_nodepth.clone(
    #in the old PFClusterAlgo this is same as barrel
    # logWeightDenominator = cms.double(0.3) 
    )

positionCalcEB_3x3_nodepth = positionCalcEB_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
    )

positionCalcEE_3x3_nodepth = positionCalcEE_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
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
