
import FWCore.ParameterSet.Config as cms

from particleFlowClusterECALTimeResolutionParameters_cfi import _timeResolutionECALBarrel, _timeResolutionECALEndcap


#until we are actually clustering across the EB/EE boundary
#it is faster to cluster EB and EE as separate
particleFlowRecHitECALWithTime = cms.EDProducer("PFRecHitProducer",

    navigator = cms.PSet(
        name = cms.string("PFRecHitECALNavigatorWithTime"),
        barrel = cms.PSet(
             noiseLevel = cms.double(0.042),   
             noiseTerm  = cms.double(27.5),
             constantTerm = cms.double(10),
             sigmaCut = cms.double(5.0),
             timeResolutionCalc = _timeResolutionECALBarrel
        ),
        endcap = cms.PSet(
             noiseLevel = cms.double(0.14),   
             noiseTerm  = cms.double(36.1),
             constantTerm = cms.double(10),
             sigmaCut = cms.double(5.0),
             timeResolutionCalc = _timeResolutionECALEndcap
        )
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
                  timingCleaning = cms.bool(False), 
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
                 timingCleaning = cms.bool(False),
                 topologicalCleaning = cms.bool(True),
                 skipTTRecoveredHits = cms.bool(True)
                 )
            )
          )
    )
          
)
