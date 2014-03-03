import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterPositionCalculators_cfi import *

topoClusterizer_ECAL = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("ECAL_BARREL"),
                 gatheringThreshold = cms.double(0.08),
                 gatheringThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("ECAL_ENDCAP"),
                 gatheringThreshold = cms.double(0.3),
                 gatheringThresholdPt = cms.double(0.0)
                 )
       ),
    useCornerCells = cms.bool(True)
    )

topoClusterizer_PS = topoClusterizer_ECAL.clone(
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("PS1"),
                 gatheringThreshold = cms.double(6e-5),
                 gatheringThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("PS2"),
                 gatheringThreshold = cms.double(6e-5),
                 gatheringThresholdPt = cms.double(0.0)
                 )
       ),    
    useCornerCells = cms.bool(False)
    )

topoClusterizer_HCAL = topoClusterizer_ECAL.clone(
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HCAL_BARREL1"),
                 gatheringThreshold = cms.double(0.8),
                 gatheringThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                 gatheringThreshold = cms.double(0.8),
                 gatheringThresholdPt = cms.double(0.0)
                 )
       )    
    )

topoClusterizer_HO = topoClusterizer_ECAL.clone(
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
                 gatheringThreshold = cms.double(0.5),
                 gatheringThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
                 gatheringThreshold = cms.double(1.0),
                 gatheringThresholdPt = cms.double(0.0)
                 )
       )    
    )

topoClusterizer_HF = topoClusterizer_ECAL.clone(
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HF_EM"),
                 gatheringThreshold = cms.double(0.8),
                 gatheringThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("HF_HAD"),
                 gatheringThreshold = cms.double(0.8),
                 gatheringThresholdPt = cms.double(0.0)
                 )
       ),
    useCornerCells = cms.bool(False)
    )

pfClusterizer_ECAL = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcECAL_3x3_nodepth,
    allCellsPositionCalc = positionCalcECAL_all_nodepth,
    positionCalcForConvergence = positionCalcECAL_all_withdepth,
    showerSigma = cms.double(1.5),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
                            cms.PSet( detector = cms.string("ECAL_BARREL"),
                                      recHitEnergyNorm = cms.double(0.08)
                                      ),
                            cms.PSet( detector = cms.string("ECAL_ENDCAP"),
                                      recHitEnergyNorm = cms.double(0.3)
                                      )
                            )
    )

pfClusterizerWithTime_ECAL = cms.PSet(
    algoName = cms.string("PFlow2DClusterizerWithTime"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcECAL_3x3_nodepth,
    allCellsPositionCalc = positionCalcECAL_all_nodepth,
    positionCalcForConvergence = positionCalcECAL_all_withdepth,
    showerSigma = cms.double(1.5),
    timeSigmaEB = cms.double(5000), 
    timeSigmaEE = cms.double(10000), 
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
                            cms.PSet( detector = cms.string("ECAL_BARREL"),
                                      recHitEnergyNorm = cms.double(0.08)
                                      ),
                            cms.PSet( detector = cms.string("ECAL_ENDCAP"),
                                      recHitEnergyNorm = cms.double(0.3)
                                      )
                            )
    )


pfClusterizer_HCAL = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcHCAL_cross_nodepth,
    allCellsPositionCalc = positionCalcHCAL_all_nodepth,
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
                            cms.PSet( detector = cms.string("HCAL_BARREL1"),
                                      recHitEnergyNorm = cms.double(0.8)
                                      ),
                            cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                                      recHitEnergyNorm = cms.double(0.8)
                                      )
                            )
    )

pfClusterizer_HO = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcHO_cross_nodepth,
    allCellsPositionCalc = positionCalcHO_all_nodepth,
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
                            cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
                                      recHitEnergyNorm = cms.double(0.5)
                                      ),
                            cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
                                      recHitEnergyNorm = cms.double(1.0)
                                      )
                            )
    )

pfClusterizer_HF = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcHF_cross_nodepth,
    allCellsPositionCalc = positionCalcHF_all_nodepth,
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
                            cms.PSet( detector = cms.string("HF_EM"),
                                      recHitEnergyNorm = cms.double(0.8)
                                      ),
                            cms.PSet( detector = cms.string("HF_HAD"),
                                      recHitEnergyNorm = cms.double(0.8)
                                      )
                            )
    )

pfClusterizer_PS = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcPS_all_nodepth,
    showerSigma = cms.double(0.3),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
                            cms.PSet( detector = cms.string("PS1"),
                                      recHitEnergyNorm = cms.double(6e-5)
                                      ),
                            cms.PSet( detector = cms.string("PS2"),
                                      recHitEnergyNorm = cms.double(6e-5)
                                      )
                            )
    )
