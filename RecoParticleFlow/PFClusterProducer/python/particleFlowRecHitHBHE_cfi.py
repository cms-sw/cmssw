import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCAL

particleFlowRecHitHBHE = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
            name = cms.string("PFRecHitHCALNavigator"),
            sigmaCut = cms.double(4.0),
            timeResolutionCalc = _timeResolutionHCAL
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
                            threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8),
                            detectorEnum = cms.int32(1)
                            ),
                        cms.PSet(
                            depth=cms.vint32(1, 2, 3, 4, 5, 6, 7),
                            threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
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

cuts2017 = particleFlowRecHitHBHE.producers[0].qualityTests[0].cuts

cuts2018 = cms.VPSet(
    cms.PSet(
        depth=cms.vint32(1, 2, 3, 4),
        threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8),
        detectorEnum = cms.int32(1)
        ),
    cms.PSet(
        depth=cms.vint32(1, 2, 3, 4, 5, 6, 7),
        threshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        detectorEnum = cms.int32(2)
        )
    )

cuts2019 = cms.VPSet(
    cms.PSet(
        depth=cms.vint32(1, 2, 3, 4),
        threshold = cms.vdouble(0.1, 0.2, 0.3, 0.3),
        detectorEnum = cms.int32(1)
        ),
    cms.PSet(
        depth=cms.vint32(1, 2, 3, 4, 5, 6, 7),
        threshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        detectorEnum = cms.int32(2)
        )
    )

cutsPhase2 = cms.VPSet(
    cms.PSet(
        depth=cms.vint32(1, 2, 3, 4),
        threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8),
        detectorEnum = cms.int32(1)
        ),
    cms.PSet(
        depth=cms.vint32(1, 2, 3, 4, 5, 6, 7),
        threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        detectorEnum = cms.int32(2)
        )
    )

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(particleFlowRecHitHBHE, cuts = cuts2018)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(particleFlowRecHitHBHE, cuts = cuts2018)

"""
# offline 2018 -- collapsed (this need PR 21842)
from Configuration.Eras.Modifier_run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toModify(particleFlowRecHitHBHE, cuts = cuts2017)
"""

# offline 2019
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowRecHitHBHE, cuts = cuts2019)

# offline phase2 restore what has been studied in the TDR
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(particleFlowRecHitHBHE, cuts = cutsPhase2)
