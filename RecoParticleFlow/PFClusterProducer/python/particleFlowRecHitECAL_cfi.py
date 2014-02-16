
import FWCore.ParameterSet.Config as cms

particleFlowRecHitECAL = cms.EDProducer("PFRecHitProducerECAL",
    # is navigation able to cross the barrel-endcap border?
    crossBarrelEndcapBorder = cms.bool(False),
    # verbosity 
    verbose = cms.untracked.bool(False),
    ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ecalRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    # cell threshold in ECAL barrel 
    thresh_Barrel = cms.double(0.08),
    # cell threshold in ECAL endcap 
    thresh_Endcap = cms.double(0.3),
    # Cleaning with timing
    timing_Cleaning = cms.bool(True),
    thresh_Cleaning_EB = cms.double(2.0),
    thresh_Cleaning_EE = cms.double(2.0),                
    # Cleaning with topology
    topological_Cleaning = cms.bool(True)
)


particleFlowRecHitECALNew = cms.EDProducer("PFRecHitProducerNew",

    navigator = cms.PSet(
        name = cms.string("PFRecHitEcalNavigator")
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFEBRecHitCreator"),
             src  = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
             isEndcap = cms.bool(False),
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
            isEndcap = cms.bool(True),
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
