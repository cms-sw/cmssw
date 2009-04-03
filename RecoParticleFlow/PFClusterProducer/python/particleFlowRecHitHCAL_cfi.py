import FWCore.ParameterSet.Config as cms

particleFlowRecHitHCAL = cms.EDFilter("PFRecHitProducerHCAL",
    # verbosity 
    verbose = cms.untracked.bool(False),
    caloTowers = cms.InputTag("towerMakerPF"),
    hcalRecHitsHBHE = cms.InputTag(""),
    # cell threshold in barrel 
    thresh_Barrel = cms.double(0.4),
    # cell threshold in HF
    thresh_HF = cms.double(0.4),
    # cell threshold in endcap 
    thresh_Endcap = cms.double(0.4),
    # Navigation in HF: 
    # False = no real clustering in HF
    # True  = do clustering in HF
    navigation_HF = cms.bool(False),
#AUGUSTE: TO BE CHECKED:
    weight_HFem = cms.double(1.429),
    weight_HFhad = cms.double(1.429)
#   weight_HFem = cms.double(1.0),
#   weight_HFhad = cms.double(1.0)
                                  
)


