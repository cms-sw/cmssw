import FWCore.ParameterSet.Config as cms

towermaker = cms.EDProducer("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2), ## GeV, Scheme B

    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(False),
    EESumThreshold = cms.double(0.45), ## GeV, Scheme B

    HOThreshold0 = cms.double(1.1),
    HOThresholdPlus1 = cms.double(1.1),
    HOThresholdMinus1 = cms.double(1.1),
    HOThresholdPlus2 = cms.double(1.1),
    HOThresholdMinus2 = cms.double(1.1),
                          
    HBThreshold = cms.double(0.9), ## GeV, Scheme B

    EBThreshold = cms.double(0.09), ## GeV, ORCA value w/o selective readout

    HcalThreshold = cms.double(-1000.0), ## GeV, -1000 means cut not used 

    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8), ## GeV, Oprimized on 10% occupancy

    EEThreshold = cms.double(0.45), ## GeV, ORCA value w/o selective readout

    HESThreshold = cms.double(1.4), ## GeV, Scheme B

    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(1.2), ## GeV, Oprimized on 10% occupancy

    HEDThreshold = cms.double(1.4), ## GeV, Scheme B

    EcutTower = cms.double(-1000.0), ## GeV, -1000 means cut not used

    ecalInputs = cms.VInputTag(cms.InputTag("ecalrechit","EcalRecHitsEB")),
    HBWeight = cms.double(1.0),

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


