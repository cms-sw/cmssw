import FWCore.ParameterSet.Config as cms

particleFlowRecHitPS = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string('PFRecHitPreshowerNavigator')
    ),
    producers = cms.VPSet(cms.PSet(
        name = cms.string('PFPSRecHitCreator'),
        qualityTests = cms.VPSet(
            cms.PSet(
                name = cms.string('PFRecHitQTestThreshold'),
                threshold = cms.double(0.0)
            ),
            cms.PSet(
                cleaningThreshold = cms.double(0.0),
                name = cms.string('PFRecHitQTestES'),
                topologicalCleaning = cms.bool(True)
            )
        ),
        src = cms.InputTag("hltEcalPreshowerRecHit","EcalRecHitsES")
    ))
)
