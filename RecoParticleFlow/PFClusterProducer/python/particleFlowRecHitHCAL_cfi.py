import FWCore.ParameterSet.Config as cms

particleFlowRecHitHCAL = cms.EDFilter("PFRecHitProducerHCAL",
    hcalRecHitsHBHE = cms.InputTag(""),
    # cell threshold in barrel 
    thresh_Barrel = cms.double(0.8),
    # verbosity 
    verbose = cms.untracked.bool(False),
    # cell threshold in endcap 
    thresh_Endcap = cms.double(0.8),
    caloTowers = cms.InputTag("towerMakerPF")
)


