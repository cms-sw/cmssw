
import FWCore.ParameterSet.Config as cms

#until we are actually clustering across the EB/EE boundary
#it is faster to cluster EB and EE as separate

particleFlowRecHitECAL = cms.EDProducer("PFRecHitProducer",

    navigator = cms.PSet(
        name = cms.string("PFRecHitECALNavigator"),
        barrel = cms.PSet( ),
        endcap = cms.PSet( )
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFEBRecHitCreator"),
             src  = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestThreshold"),
                  threshold = cms.double(0.08)
                  ),
                  cms.PSet(
                  name = cms.string("PFRecHitQTestECAL"),
                  cleaningThreshold = cms.double(2.0),
                  timingCleaning = cms.bool(True),
                  topologicalCleaning = cms.bool(True),
                  skipTTRecoveredHits = cms.bool(True)
                  )
             )
           ),
          cms.PSet(
            name = cms.string("PFEERecHitCreator"),
            src  = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            qualityTests = cms.VPSet(
                 cms.PSet(
                 name = cms.string("PFRecHitQTestThreshold"),
                 threshold = cms.double(0.3)
                 ),
                 cms.PSet(
                 name = cms.string("PFRecHitQTestECAL"),
                 cleaningThreshold = cms.double(2.0),
                 timingCleaning = cms.bool(True),
                 topologicalCleaning = cms.bool(True),
                 skipTTRecoveredHits = cms.bool(True)
                 )
            )
          )
    )
          
)
