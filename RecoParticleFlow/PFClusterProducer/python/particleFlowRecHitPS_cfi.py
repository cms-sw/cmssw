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


