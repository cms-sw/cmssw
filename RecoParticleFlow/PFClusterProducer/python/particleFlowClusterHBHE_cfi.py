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
                        seedingThreshold = cms.double(1.0),
                        seedingThresholdPt = cms.double(0.0)
                        ),
              cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                        seedingThreshold = cms.double(1.1),
                        seedingThresholdPt = cms.double(0.0)
                        )
              ),
        nNeighbours = cms.int32(4)
    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string("Basic2DGenericTopoClusterizer"),    
        thresholdsByDetector = cms.VPSet(
        cms.PSet( detector = cms.string("HCAL_BARREL1"),
                  gatheringThreshold = cms.double(0.8),
                  gatheringThresholdPt = cms.double(0.0)
                  ),
        cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                  gatheringThreshold = cms.double(0.8),
                  gatheringThresholdPt = cms.double(0.0)
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
                     recHitEnergyNorm = cms.double(0.8)
                     ),
           cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                     recHitEnergyNorm = cms.double(0.8)
                     )
           )
    ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

