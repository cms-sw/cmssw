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
    thresh_Cleaning_EE = cms.double(1E9), # no time clean in EE               
    # Cleaning with topology
    topological_Cleaning = cms.bool(True)
)


