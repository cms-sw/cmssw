import FWCore.ParameterSet.Config as cms

towerMakerPF = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(-1000.0),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(True),
    EESumThreshold = cms.double(-1000.0),
    HOThreshold = cms.double(999999.0),
    HBThreshold = cms.double(0.0),
    EBThreshold = cms.double(999999.0),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(False),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(999999.0),
    EEThreshold = cms.double(999999.0),
    HESThreshold = cms.double(0.0),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(999999.0),
    HEDThreshold = cms.double(0.0),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)


