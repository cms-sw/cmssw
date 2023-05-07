import FWCore.ParameterSet.Config as cms

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
#updated HB RecHit threshold for 2022 rereco
_thresholdsHBphase1_2022_rereco = cms.vdouble(0.25, 0.25, 0.3, 0.3)

particleFlowClusterHCAL = cms.EDProducer('PFMultiDepthClusterProducer',
       clustersSource = cms.InputTag("particleFlowClusterHBHE"),
       pfClusterBuilder =cms.PSet(
           algoName = cms.string("PFMultiDepthClusterizer"),
           nSigmaEta = cms.double(2.),
           nSigmaPhi = cms.double(2.),
           #pf clustering parameters
           minFractionToKeep = cms.double(1e-7),
           allCellsPositionCalc = cms.PSet(
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
           )
       ),
       positionReCalc = cms.PSet(),
       energyCorrector = cms.PSet()
)

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
(run2_HE_2018 & ~run2_HECollapse_2018).toModify(particleFlowClusterHCAL,
    pfClusterBuilder = dict(
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {1 : dict(logWeightDenominator = _thresholdsHEphase1) } ),
    ),
)

# offline 2021
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowClusterHCAL,
    pfClusterBuilder = dict(
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {0 : dict(logWeightDenominator = _thresholdsHBphase1) } ),
    ),
)

# offline 2022 rereco
from Configuration.Eras.Modifier_run3_egamma_2022_rereco_cff import run3_egamma_2022_rereco
run3_egamma_2022_rereco.toModify(particleFlowClusterHCAL,
    pfClusterBuilder = dict(
        allCellsPositionCalc = dict(logWeightDenominatorByDetector = {0 : dict(logWeightDenominator = _thresholdsHBphase1_2022_rereco) } ),
    ),
)

# HCALonly WF
particleFlowClusterHCALOnly = particleFlowClusterHCAL.clone(
    clustersSource = "particleFlowClusterHBHEOnly"
)
