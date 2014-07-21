
import FWCore.ParameterSet.Config as cms

from particleFlowCaloResolution_cfi import _timeResolutionECALBarrel, _timeResolutionECALEndcap


#until we are actually clustering across the EB/EE boundary
#it is faster to cluster EB and EE as separate
particleFlowRecHitECALWithTime = cms.EDProducer("PFRecHitProducer",

    navigator = cms.PSet(
        name = cms.string("PFRecHitECALNavigatorWithTime"),
        barrel = cms.PSet(
             sigmaCut = cms.double(5.0),
             timeResolutionCalc = _timeResolutionECALBarrel # this overrides the previous if it exists
        ),
        endcap = cms.PSet(
             timeResolutionCalc = _timeResolutionECALEndcap
        )
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFEBRecHitCreatorMaxSample"),
             src  = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestThreshold"),
                  threshold = cms.double(0.08)
                  ),
                  cms.PSet(
                  name = cms.string("PFRecHitQTestECAL"),
                  cleaningThreshold = cms.double(2.0),
                  timingCleaning = cms.bool(False), 
                  topologicalCleaning = cms.bool(True),
                  skipTTRecoveredHits = cms.bool(True)
                  )
             )
           ),
          cms.PSet(
            name = cms.string("PFEERecHitCreatorMaxSample"),
            src  = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            qualityTests = cms.VPSet(
                 cms.PSet(
                 name = cms.string("PFRecHitQTestThreshold"),
                 threshold = cms.double(0.3)
                 ),
                 cms.PSet(
                 name = cms.string("PFRecHitQTestECAL"),
                 cleaningThreshold = cms.double(2.0),
                 timingCleaning = cms.bool(False),
                 topologicalCleaning = cms.bool(True),
                 skipTTRecoveredHits = cms.bool(True)
                 )
            )
          )
    )
          
)
