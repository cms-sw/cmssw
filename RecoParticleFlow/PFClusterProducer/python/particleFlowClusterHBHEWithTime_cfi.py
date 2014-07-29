import FWCore.ParameterSet.Config as cms
from particleFlowCaloResolution_cfi import _timeResolutionHCAL

#### PF CLUSTER HCAL ####
particleFlowClusterHBHEWithTime = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHBHE"),
    recHitCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string("LocalMaximumSeedFinder"),
        thresholdsByDetector = cms.VPSet(
              cms.PSet( detector = cms.string("HCAL_BARREL1"),
                        seedingThreshold = cms.double(0.5),
                        seedingThresholdPt = cms.double(0.0)
                        ),
              cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                        seedingThreshold = cms.double(0.5),
                        seedingThresholdPt = cms.double(0.0)
                        )
              ),
        nNeighbours = cms.int32(4)
    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string("Basic2DGenericTopoClusterizer"),    
        thresholdsByDetector = cms.VPSet(
        cms.PSet( detector = cms.string("HCAL_BARREL1"),
                  gatheringThreshold = cms.double(0.3),
                  gatheringThresholdPt = cms.double(0.0)
                  ),
        cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                  gatheringThreshold = cms.double(0.3),
                  gatheringThresholdPt = cms.double(0.0)
                  )
        ),
        useCornerCells = cms.bool(True)
    ),
    
    pfClusterBuilder = cms.PSet(
           algoName = cms.string("PFlow2DClusterizerWithTime"),
           #pf clustering parameters
           minFractionToKeep = cms.double(1e-7),
           positionCalc = cms.PSet(
                 algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                 minFractionInCalc = cms.double(1e-9),    
                 posCalcNCrystals = cms.int32(5),
                 logWeightDenominator = cms.double(0.3),#same as gathering threshold
                 minAllowedNormalization = cms.double(1e-9)
           ),
           allCellsPositionCalc =cms.PSet(
                 algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                 minFractionInCalc = cms.double(1e-9),    
                 posCalcNCrystals = cms.int32(-1),
                 logWeightDenominator = cms.double(0.3),#same as gathering threshold
                 minAllowedNormalization = cms.double(1e-9)
           ),
           
           showerSigma = cms.double(10.0),
           timeSigmaEB = cms.double(2),
           timeSigmaEE = cms.double(2), 
           maxNSigmaTime = cms.double(10.), # Maximum number of sigmas in time 
           minChi2Prob = cms.double(0.), # Minimum chi2 probability (ignored if 0)
           stoppingTolerance = cms.double(1e-8),
           maxIterations = cms.uint32(50),
           excludeOtherSeeds = cms.bool(True),
           minFracTot = cms.double(1e-20), ## numerical stabilization
           clusterTimeResFromSeed = cms.bool(False),
           recHitEnergyNorms = cms.VPSet(
           cms.PSet( detector = cms.string("HCAL_BARREL1"),
                     recHitEnergyNorm = cms.double(0.5)
                     ),
           cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                     recHitEnergyNorm = cms.double(0.5)
                     )
           ),
    timeResolutionCalcBarrel = _timeResolutionHCAL,
    timeResolutionCalcEndcap = _timeResolutionHCAL
    ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

