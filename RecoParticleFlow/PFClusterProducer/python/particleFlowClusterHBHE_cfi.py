import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)

#### PF CLUSTER HCAL ####
particleFlowClusterHBHE = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHBHE"),
    recHitCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string("LocalMaximumSeedFinder"),
        thresholdsByDetector = cms.VPSet(
              cms.PSet( detector = cms.string("HCAL_BARREL1"),
                        depths = cms.vint32(1, 2, 3, 4),
                        seedingThreshold = cms.vdouble(1.0, 1.0, 1.0, 1.0),
                        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
                        ),
              cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                        seedingThreshold = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1),
                        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        )
              ),
        nNeighbours = cms.int32(4)
    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string("Basic2DGenericTopoClusterizer"),    
        thresholdsByDetector = cms.VPSet(
        cms.PSet( detector = cms.string("HCAL_BARREL1"),
                  depths = cms.vint32(1, 2, 3, 4),
                  gatheringThreshold = _thresholdsHB,
                  gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
                  ),
        cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                  depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                  gatheringThreshold = _thresholdsHE,
                  gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                  )
        ),
        useCornerCells = cms.bool(True)
    ),
    
    pfClusterBuilder = cms.PSet(
           algoName = cms.string("Basic2DGenericPFlowClusterizer"),
           #pf clustering parameters
           minFractionToKeep = cms.double(1e-7),
           positionCalc = cms.PSet(
                 algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                 minFractionInCalc = cms.double(1e-9),    
                 posCalcNCrystals = cms.int32(5),
#                 logWeightDenominator = cms.double(0.8),#same as gathering threshold
                 logWeightDenominatorByDetector = cms.VPSet(
                       cms.PSet( detector = cms.string("HCAL_BARREL1"),
                                 depths = cms.vint32(1, 2, 3, 4),
                                 logWeightDenominator = _thresholdsHB,
                                 ),
                       cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                                 depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                                 logWeightDenominator = _thresholdsHE,
                                 )
                       ),
                 minAllowedNormalization = cms.double(1e-9)
           ),
           allCellsPositionCalc =cms.PSet(
                 algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                 minFractionInCalc = cms.double(1e-9),    
                 posCalcNCrystals = cms.int32(-1),
##                 logWeightDenominator = cms.double(0.8),#same as gathering threshold
                 logWeightDenominatorByDetector = cms.VPSet(
                       cms.PSet( detector = cms.string("HCAL_BARREL1"),
                                 depths = cms.vint32(1, 2, 3, 4),
                                 logWeightDenominator = _thresholdsHB,
                                 ),
                       cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                                 depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                                 logWeightDenominator = _thresholdsHE,
                                 )
                       ),
                 minAllowedNormalization = cms.double(1e-9)
           ),
           

           timeSigmaEB = cms.double(10.),
           timeSigmaEE = cms.double(10.),
           maxNSigmaTime = cms.double(10.),
           minChi2Prob = cms.double(0.),
           clusterTimeResFromSeed = cms.bool(False),
           timeResolutionCalcBarrel = _timeResolutionHCALMaxSample,
           timeResolutionCalcEndcap = _timeResolutionHCALMaxSample,
           showerSigma = cms.double(10.0),
           stoppingTolerance = cms.double(1e-8),
           maxIterations = cms.uint32(50),
           excludeOtherSeeds = cms.bool(True),
           minFracTot = cms.double(1e-20), ## numerical stabilization
           recHitEnergyNorms = cms.VPSet(
            cms.PSet( detector = cms.string("HCAL_BARREL1"),
                      depths = cms.vint32(1, 2, 3, 4),
                      recHitEnergyNorm = _thresholdsHB,
                      ),
            cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                      depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                      recHitEnergyNorm = _thresholdsHE,
                      )
            )
    ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

#####

seedFinderThresholdsByDetector2017 = particleFlowClusterHBHE.seedFinder.thresholdsByDetector

seedFinderThresholdsByDetector2018 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        seedingThreshold = cms.vdouble(1.0, 1.0, 1.0, 1.0),
        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        seedingThreshold = cms.vdouble(0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275),
        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
)

seedFinderThresholdsByDetector2019 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        seedingThreshold = cms.vdouble(0.125, 0.25, 0.25, 0.25),
        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        seedingThreshold = cms.vdouble(0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275),
        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
)

seedFinderThresholdsByDetectorPhase2 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        seedingThreshold = cms.vdouble(1.0, 1.0, 1.0, 1.0),
        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        seedingThreshold = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1),
        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    )

#######################

initialClusteringStepThresholdsByDetector2017 = particleFlowClusterHBHE.initialClusteringStep.thresholdsByDetector

initialClusteringStepThresholdsByDetector2018 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        gatheringThreshold = _thresholdsHB,
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        gatheringThreshold = _thresholdsHEphase1,
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
)

initialClusteringStepThresholdsByDetector2019 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        gatheringThreshold = _thresholdsHBphase1,
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        gatheringThreshold = _thresholdsHEphase1,
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
)

initialClusteringStepThresholdsByDetectorPhase2 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        gatheringThreshold = _thresholdsHB,
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        gatheringThreshold = _thresholdsHE,
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    )

#######################

recHitEnergyNorms2017 = particleFlowClusterHBHE.pfClusterBuilder.recHitEnergyNorms

recHitEnergyNorms2018 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        recHitEnergyNorm = _thresholdsHB,
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        recHitEnergyNorm = _thresholdsHEphase1,
        )
)

recHitEnergyNorms2019 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        recHitEnergyNorm = _thresholdsHBphase1,
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        recHitEnergyNorm = _thresholdsHEphase1,
        )
    )

recHitEnergyNormsPhase2 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        recHitEnergyNorm = _thresholdsHB,
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        recHitEnergyNorm = _thresholdsHE,
        )
    )

#######################

logWeightDenominatorByDetector2017= particleFlowClusterHBHE.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector

logWeightDenominatorByDetector2018 = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL1"),
              depths = cms.vint32(1, 2, 3, 4),
              logWeightDenominator = _thresholdsHB
              ),
    cms.PSet( detector = cms.string("HCAL_ENDCAP"),
              depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
              logWeightDenominator = _thresholdsHEphase1,
              )
    )

logWeightDenominatorByDetector2019 = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL1"),
              depths = cms.vint32(1, 2, 3, 4),
              logWeightDenominator = _thresholdsHBphase1,
              ),
    cms.PSet( detector = cms.string("HCAL_ENDCAP"),
              depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
              logWeightDenominator = _thresholdsHEphase1,
              )
    )

logWeightDenominatorByDetectorPhase2 = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL1"),
              depths = cms.vint32(1, 2, 3, 4),
              logWeightDenominator = _thresholdsHB,
              ),
    cms.PSet( detector = cms.string("HCAL_ENDCAP"),
              depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
              logWeightDenominator = _thresholdsHE,
              )
    )

#######################

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder, recHitEnergyNorms = recHitEnergyNorms2018)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2018)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2018)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder.positionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2018)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2018)


from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder, recHitEnergyNorms = recHitEnergyNorms2018)
run2_HE_2018.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2018)
run2_HE_2018.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2018)
run2_HE_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder.positionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2018)
run2_HE_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2018)

# offline 2018 -- collapsed
run2_HECollapse_2018 =  cms.Modifier()
run2_HECollapse_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder, recHitEnergyNorms = recHitEnergyNorms2017)
run2_HECollapse_2018.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2017)
run2_HECollapse_2018.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2017)
run2_HECollapse_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder.positionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2017)
run2_HECollapse_2018.toModify(particleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2017)


# offline 2019
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowClusterHBHE.pfClusterBuilder, recHitEnergyNorms = recHitEnergyNorms2019)
run3_HB.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2019)
run3_HB.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2019)
run3_HB.toModify(particleFlowClusterHBHE.pfClusterBuilder.positionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2019)
run3_HB.toModify(particleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2019)

# offline phase2 restore what has been studied in the TDR
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(particleFlowClusterHBHE.pfClusterBuilder, recHitEnergyNorms = recHitEnergyNormsPhase2)
phase2_hcal.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetectorPhase2)
phase2_hcal.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetectorPhase2)
phase2_hcal.toModify(particleFlowClusterHBHE.pfClusterBuilder.positionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetectorPhase2)
phase2_hcal.toModify(particleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetectorPhase2)
