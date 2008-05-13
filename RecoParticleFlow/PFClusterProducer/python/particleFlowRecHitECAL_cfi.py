import FWCore.ParameterSet.Config as cms

particleFlowRecHitECAL = cms.EDFilter("PFRecHitProducerECAL",
    # is navigation able to cross the barrel-endcap border?
    crossBarrelEndcapBorder = cms.bool(False),
    # verbosity 
    verbose = cms.untracked.bool(False),
    ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ecalRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    # cell threshold in ECAL barrel 
    thresh_Barrel = cms.double(0.08),
    # cell threshold in ECAL endcap 
    thresh_Endcap = cms.double(0.3)
)


