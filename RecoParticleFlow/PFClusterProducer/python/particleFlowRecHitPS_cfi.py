import FWCore.ParameterSet.Config as cms

particleFlowRecHitPS = cms.EDProducer("PFRecHitProducerPS",
    # cell threshold in barrel 
    thresh_Barrel = cms.double(7e-06),
    ecalRecHitsES = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    # cell threshold in endcap 
    thresh_Endcap = cms.double(7e-06),
    # verbosity 
    verbose = cms.untracked.bool(False)
)




particleFlowRecHitPSNew = cms.EDProducer("PFRecHitProducerNew",
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
                  threshold = cms.double(7e-6)
                  )
             )
           )
    )

)

