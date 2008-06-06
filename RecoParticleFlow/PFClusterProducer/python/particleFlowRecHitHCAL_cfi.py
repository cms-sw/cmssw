import FWCore.ParameterSet.Config as cms

particleFlowRecHitHCAL = cms.EDFilter("PFRecHitProducerHCAL",
    # verbosity 
    verbose = cms.untracked.bool(False),
    caloTowers = cms.InputTag("towerMakerPF"),
    hcalRecHitsHBHE = cms.InputTag(""),
    # cell threshold in barrel 
    thresh_Barrel = cms.double(0.8),
    # cell threshold in HF
    thresh_HF = cms.double(0.0),
    # cell threshold in endcap 
    thresh_Endcap = cms.double(0.8)
)


