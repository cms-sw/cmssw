import FWCore.ParameterSet.Config as cms

towermaker = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(False),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(1.2),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalrechit","EcalRecHitsEB")),
    HBWeight = cms.double(1.0)
)


