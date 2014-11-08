import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 

#seeding
_localmaxseeds_HGCEE = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HGC_ECAL"),
              #seeding threshold converted to GeV from keV
              seedingThreshold = cms.double(1e-6*1.75*55.1),
              seedingThresholdPt = cms.double(0.0)
              )
    ),
    nNeighbours = cms.int32(8)
)

_positionCalcHGCEE_onelayer = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(1e-6*0.25*55.1), # use ADC value 0.25*MIP
    minAllowedNormalization = cms.double(1e-9)
    )

_positionCalcHGCEE_pca = cms.PSet(
    algoName = cms.string("Cluster3DPCACalculator"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(1e-6*1.0*55.1), # use 1 MIP
    minAllowedNormalization = cms.double(1e-9)
    )

_fromScratchHGCClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("HGCClusterizer"), 
    thresholdsByDetector = cms.VPSet( ),
    positionCalcInLayer = _positionCalcHGCEE_onelayer,
    positionCalcPCA = _positionCalcHGCEE_pca,
    moliereRadii = cms.PSet( HGC_ECAL = cms.double(2.9)  ) #cm
)

#weights for layers from P.Silva (24 October 2014)
## this is for V5!
weight_vec = [0.080]
weight_vec.extend([0.62 for x in range(9)])
weight_vec.extend([0.81 for x in range(9)])
weight_vec.extend([1.19 for x in range(8)])

# MIP effective to 1.0/GeV (from fit to data of P. Silva)
#f(x) = a/(1-exp(-bx - c))
# x = cosh(eta)
# a = 82.8
# b = 1e6
# c = 1e6

_HGCEE_ElectronEnergy = cms.PSet(
    algoName = cms.string("HGCEEElectronEnergyCalibrator"),
    weights = cms.vdouble(weight_vec),
    effMip_to_InverseGeV_a = cms.double(82.8),
    effMip_to_InverseGeV_b = cms.double(1e6),
    effMip_to_InverseGeV_c = cms.double(1e6),
    MipValueInGeV = cms.double(55.1*1e-6)
)

particleFlowClusterHGCEE = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCEE"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _localmaxseeds_HGCEE,
    initialClusteringStep = _fromScratchHGCClusterizer_HGCEE,
    pfClusterBuilder = cms.PSet( ), #_arborClusterizer_HGCEE,
    positionReCalc = cms.PSet( ), #_simplePosCalcHGCEE,
    energyCorrector = _HGCEE_ElectronEnergy
)

