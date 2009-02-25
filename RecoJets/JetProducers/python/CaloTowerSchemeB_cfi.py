import FWCore.ParameterSet.Config as cms

towerMaker = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    HF2Weight = cms.double(1.0),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    EESumThreshold = cms.double(0.45),
    # Energy threshold for HO cell inclusion [GeV]
    HOThreshold0 = cms.double(1.1),
    HOThresholdPlus1 = cms.double(1.1),
    HOThresholdMinus1 = cms.double(1.1),
    HOThresholdPlus2 = cms.double(1.1),
    HOThresholdMinus2 = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    HF1Threshold = cms.double(1.2),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(False),
    HESWeight = cms.double(1.0),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    EBThreshold = cms.double(0.09),
    hbheInput = cms.InputTag("hbhereco"),
    HcalThreshold = cms.double(-1000.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("horeco"),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    AllowMissingInputs = cms.untracked.bool(False),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0),
    MomConstrMethod = cms.int32(1),
    MomHBDepth = cms.double(0.2),
    MomHEDepth = cms.double(0.4),
    MomEBDepth = cms.double(0.3),
    MomEEDepth = cms.double(0.0),

# add new parameters for handling of anomalous cells
# EXAMPLE 
# 
    # acceptable severity level
    HcalAcceptSeverityLevel = cms.uint32(999),
    EcalAcceptSeverityLevel = cms.uint32(1),

    # use of recovered hits
    UseHcalRecoveredHits = cms.bool(True),
    UseEcalRecoveredHits = cms.bool(True)
)
