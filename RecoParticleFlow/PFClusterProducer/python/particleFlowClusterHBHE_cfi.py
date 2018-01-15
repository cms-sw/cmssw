import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample

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
                  gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8),
                  gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
                  ),
        cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                  depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                  gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
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
                 logWeightDenominator = cms.double(0.8),#same as gathering threshold
                 minAllowedNormalization = cms.double(1e-9)
           ),
           allCellsPositionCalc =cms.PSet(
                 algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                 minFractionInCalc = cms.double(1e-9),    
                 posCalcNCrystals = cms.int32(-1),
                 logWeightDenominator = cms.double(0.8),#same as gathering threshold
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
                      recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8),
                      ),
            cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                      depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                      recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
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
        gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8),
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        gatheringThreshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
)

initialClusteringStepThresholdsByDetector2019 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        gatheringThreshold = cms.vdouble(0.1, 0.2, 0.3, 0.3),
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        gatheringThreshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
)

initialClusteringStepThresholdsByDetectorPhase2 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8),
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    )

#######################

recHitEnergyNorms2017 = particleFlowClusterHBHE.pfClusterBuilder.recHitEnergyNorms

recHitEnergyNorms2018 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8),
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        recHitEnergyNorm = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        )
)

recHitEnergyNorms2019 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        recHitEnergyNorm = cms.vdouble(0.1, 0.2, 0.3, 0.3),
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        recHitEnergyNorm = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        )
    )

recHitEnergyNormsPhase2 = cms.VPSet(
    cms.PSet(
        detector = cms.string("HCAL_BARREL1"),
        depths = cms.vint32(1, 2, 3, 4),
        recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8),
        ),
    cms.PSet(
        detector = cms.string("HCAL_ENDCAP"),
        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
        recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        )
    )

#######################

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(particleFlowClusterHBHE, recHitEnergyNorms = recHitEnergyNorms2018)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2018)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2018)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(particleFlowClusterHBHE, recHitEnergyNorms = recHitEnergyNorms2018)
run2_HE_2018.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2018)
run2_HE_2018.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2018)

"""
# offline 2018 -- collapsed (this need PR 21842)
from Configuration.Eras.Modifier_run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toModify(particleFlowClusterHBHE, recHitEnergyNorms = recHitEnergyNorms2017)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2017)
run2_HCAL_2018.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2017)
"""

# offline 2019
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowClusterHBHE, recHitEnergyNorms = recHitEnergyNorms2019)
run3_HB.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetector2019)
run3_HB.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetector2019)

# offline phase2 restore what has been studied in the TDR
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(particleFlowClusterHBHE, recHitEnergyNorms = recHitEnergyNormsPhase2)
phase2_hcal.toModify(particleFlowClusterHBHE.seedFinder, thresholdsByDetector = seedFinderThresholdsByDetectorPhase2)
phase2_hcal.toModify(particleFlowClusterHBHE.initialClusteringStep, thresholdsByDetector = initialClusteringStepThresholdsByDetectorPhase2)
