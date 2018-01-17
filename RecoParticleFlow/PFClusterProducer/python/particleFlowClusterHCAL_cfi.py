import FWCore.ParameterSet.Config as cms

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0., 0.2)

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


logWeightDenominatorByDetector2017= particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector

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

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2018)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2018)

# offline 2018 -- collapsed
run2_HECollapse_2018 =  cms.Modifier()
run2_HECollapse_2018.toModify(particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2017)

# offline 2019
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetector2019)

# offline phase2 restore what has been studied in the TDR
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc, logWeightDenominatorByDetector= logWeightDenominatorByDetectorPhase2)
