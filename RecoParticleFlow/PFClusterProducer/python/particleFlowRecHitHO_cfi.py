import FWCore.ParameterSet.Config as cms
particleFlowRecHitHO = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string("PFRecHitHCALNavigator")
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHORecHitCreator"),
             src  = cms.InputTag("horeco",""),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestHOThreshold"),
                  threshold_ring0 = cms.double(0.4),
                  threshold_ring12 = cms.double(1.0)
                  ),
                  cms.PSet(
                  name = cms.string("PFRecHitQTestHCALChannel"),
                  maxSeverity = cms.int32(11)
                  )
                  

             )
           )
    )

)

