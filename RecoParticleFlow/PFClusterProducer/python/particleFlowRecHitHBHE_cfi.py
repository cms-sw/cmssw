import FWCore.ParameterSet.Config as cms

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
#updated HB RecHit threshold for 2022 rereco
_thresholdsHBphase1_2022_rereco = cms.vdouble(0.25, 0.25, 0.3, 0.3)

particleFlowRecHitHBHE = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
            name = cms.string("PFRecHitHCALDenseIdNavigator"),
            hcalEnums = cms.vint32(1,2)
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHBHERecHitCreator"),
             src  = cms.InputTag("hbhereco",""),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestHCALThresholdVsDepth"),
                  cuts = cms.VPSet(
                        cms.PSet(
                            depth=cms.vint32(1, 2, 3, 4),
                            threshold = _thresholdsHB,
                            detectorEnum = cms.int32(1)
                            ),
                        cms.PSet(
                            depth=cms.vint32(1, 2, 3, 4, 5, 6, 7),
                            threshold = _thresholdsHE,
                            detectorEnum = cms.int32(2)
                            )
                        )
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALChannel"),
                      maxSeverities      = cms.vint32(11),
                      cleaningThresholds = cms.vdouble(0.0),
                      flags              = cms.vstring('Standard')
                  )
             )
           ),
    )
)

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
(run2_HE_2018 & ~run2_HECollapse_2018).toModify(particleFlowRecHitHBHE,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {1 : dict(threshold = _thresholdsHEphase1) } ) } ) },
)

# offline 2021
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowRecHitHBHE,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {0 : dict(threshold = _thresholdsHBphase1) } ) } ) },
)

from Configuration.Eras.Modifier_run3_egamma_2022_rereco_cff import run3_egamma_2022_rereco
run3_egamma_2022_rereco.toModify(particleFlowRecHitHBHE,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {0 : dict(threshold = _thresholdsHBphase1_2022_rereco) } ) } ) },
)

# HCALonly WF
particleFlowRecHitHBHEOnly = particleFlowRecHitHBHE.clone(
    producers = { 0: dict(src = "hbheprereco") }
)
run3_HB.toModify(particleFlowRecHitHBHEOnly,
    producers = { 0: dict(src = "hbhereco") }
)
