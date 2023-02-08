import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
_seedingThresholdsHB = cms.vdouble(1.0, 1.0, 1.0, 1.0)
_seedingThresholdsHE = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1)
_seedingThresholdsHBphase1 = cms.vdouble(0.125, 0.25, 0.35, 0.35)
_seedingThresholdsHEphase1 = cms.vdouble(0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275)

_module_config = cms.PSet(
    recHitsSource = cms.InputTag("particleFlowRecHitHBHE"),
    recHitCleaners = cms.VPSet(),
    seedCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string("LocalMaximumSeedFinder"),
        thresholdsByDetector = cms.VPSet(
              cms.PSet( detector = cms.string("HCAL_BARREL1"),
                        depths = cms.vint32(1, 2, 3, 4),
                        seedingThreshold = _seedingThresholdsHB,
                        seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)
                        ),
              cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                        depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
                        seedingThreshold = _seedingThresholdsHE,
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
_module_config_cuda = _module_config.clone()

#### PF CLUSTER HCAL ####
_particleFlowClusterHBHE_cpu = cms.EDProducer("PFClusterProducer", _module_config.clone())
_particleFlowClusterHBHE_cuda = cms.EDProducer("PFClusterProducerCudaHCAL", _module_config.clone())
_particleFlowClusterHBHE_cuda.PFRecHitsLabelIn = cms.InputTag("particleFlowRecHitHBHE","")
_particleFlowClusterHBHE_cuda.produceLegacy = cms.bool(True)
_particleFlowClusterHBHE_cuda.produceSoA = cms.bool(True)

#####

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
(run2_HE_2018 & ~run2_HECollapse_2018).toModify(_particleFlowClusterHBHE_cpu,
    seedFinder = dict(thresholdsByDetector = {1 : dict(seedingThreshold = _seedingThresholdsHEphase1) } ),
    initialClusteringStep = dict(thresholdsByDetector = {1 : dict(gatheringThreshold = _thresholdsHEphase1) } ),
    pfClusterBuilder = dict(
        recHitEnergyNorms = {1 : dict(recHitEnergyNorm = _thresholdsHEphase1) },
        positionCalc = dict(logWeightDenominatorByDetector = {1 : dict(logWeightDenominator = _thresholdsHEphase1) } ),
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {1 : dict(logWeightDenominator = _thresholdsHEphase1) } ),
    ),
)
(run2_HE_2018 & ~run2_HECollapse_2018).toModify(_particleFlowClusterHBHE_cuda,
    seedFinder = dict(thresholdsByDetector = {1 : dict(seedingThreshold = _seedingThresholdsHEphase1) } ),
    initialClusteringStep = dict(thresholdsByDetector = {1 : dict(gatheringThreshold = _thresholdsHEphase1) } ),
    pfClusterBuilder = dict(
        recHitEnergyNorms = {1 : dict(recHitEnergyNorm = _thresholdsHEphase1) },
        positionCalc = dict(logWeightDenominatorByDetector = {1 : dict(logWeightDenominator = _thresholdsHEphase1) } ),
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {1 : dict(logWeightDenominator = _thresholdsHEphase1) } ),
    ),
)

# offline 2019
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(_particleFlowClusterHBHE_cpu,
    seedFinder = dict(thresholdsByDetector = {0 : dict(seedingThreshold = _seedingThresholdsHBphase1) } ),
    initialClusteringStep = dict(thresholdsByDetector = {0 : dict(gatheringThreshold = _thresholdsHBphase1) } ),
    pfClusterBuilder = dict(
        recHitEnergyNorms = {0 : dict(recHitEnergyNorm = _thresholdsHBphase1) },
        positionCalc = dict(logWeightDenominatorByDetector = {0 : dict(logWeightDenominator = _thresholdsHBphase1) } ),
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {0 : dict(logWeightDenominator = _thresholdsHBphase1) } ),
    ),
)
run3_HB.toModify(_particleFlowClusterHBHE_cuda,
    seedFinder = dict(thresholdsByDetector = {0 : dict(seedingThreshold = _seedingThresholdsHBphase1) } ),
    initialClusteringStep = dict(thresholdsByDetector = {0 : dict(gatheringThreshold = _thresholdsHBphase1) } ),
    pfClusterBuilder = dict(
        recHitEnergyNorms = {0 : dict(recHitEnergyNorm = _thresholdsHBphase1) },
        positionCalc = dict(logWeightDenominatorByDetector = {0 : dict(logWeightDenominator = _thresholdsHBphase1) } ),
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {0 : dict(logWeightDenominator = _thresholdsHBphase1) } ),
    ),
)

# HCALonly WF
# particleFlowClusterHBHEOnly = _particleFlowClusterHBHE_cpu.clone(
#     recHitsSource = "particleFlowRecHitHBHEOnly"
# )
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
particleFlowClusterHBHEOnly = SwitchProducerCUDA(
    cpu = _particleFlowClusterHBHE_cpu.clone(
        recHitsSource = "particleFlowRecHitHBHEOnly@cpu"
    )
)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(particleFlowClusterHBHEOnly,
    cuda = _particleFlowClusterHBHE_cuda.clone(
        recHitsSource = "particleFlowRecHitHBHEOnly@cuda",
        PFRecHitsLabelIn = "particleFlowRecHitHBHEOnly@cuda"
    )
)

#
#from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
particleFlowClusterHBHE = SwitchProducerCUDA(
    cpu = _particleFlowClusterHBHE_cpu.clone()
)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(particleFlowClusterHBHE, 
    cuda = _particleFlowClusterHBHE_cuda.clone()
)         
