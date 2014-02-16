import FWCore.ParameterSet.Config as cms

topoClusterizer_EB = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    #topo clustering parameters
    gatheringThreshold = cms.double(0.08),
    gatheringThresholdPt = cms.double(0.0),
    useCornerCells = cms.bool(True)
    )

topoClusterizer_EE = topoClusterizer_EB.clone(
    gatheringThreshold = cms.double(0.3),
    gatheringThresholdPt = cms.double(0.0)
    )

pfClusterizer_EB = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcEB_3x3_nodepth,
    allCellsPositionCalc = positionCalcEB_all_nodepth,
    positionCalcForConvergence = positionCalcECAL_all_withdepth,
    showerSigma = cms.double(1.5),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True)
    )

pfClusterizer_EE = pfClusterizer_EB.clone(   
    #positionCalc = positionCalcEE_3x3_nodepth,
    #allCellsPositionCalc = positionCalcEE_all_nodepth    
    )
