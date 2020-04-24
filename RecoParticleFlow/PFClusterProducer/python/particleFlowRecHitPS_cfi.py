import FWCore.ParameterSet.Config as cms

particleFlowRecHitPS = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string("PFRecHitPreshowerNavigator")
    ),
    producers = cms.VPSet(
        cms.PSet(
            name = cms.string("PFPSRecHitCreator"),
            src  = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
            qualityTests = cms.VPSet(
                cms.PSet(
                    name = cms.string("PFRecHitQTestThreshold"),
                    threshold = cms.double(0.)
                    ),
                cms.PSet(
                    name = cms.string("PFRecHitQTestES"),
                    cleaningThreshold = cms.double(0.),
                    topologicalCleaning = cms.bool(True)
                    )
                )
            )
        )
                                      )

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(
    particleFlowRecHitPS.producers[0], 
    src = "reducedEcalRecHitsES"
)
