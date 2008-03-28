import FWCore.ParameterSet.Config as cms

particleFlowRecHitECAL = cms.EDFilter("PFRecHitProducerECAL",
    # verbosity 
    verbose = cms.untracked.bool(False),
    ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    # cell threshold in ECAL barrel 
    thresh_Barrel = cms.double(0.08),
    ecalRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    # cell threshold in ECAL endcap 
    thresh_Endcap = cms.double(0.3)
)


