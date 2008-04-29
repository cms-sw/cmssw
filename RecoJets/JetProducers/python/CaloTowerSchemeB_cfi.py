import FWCore.ParameterSet.Config as cms

towerMaker = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2), ## GeV, Scheme B

    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(False),
    EESumThreshold = cms.double(0.45), ## GeV, Scheme B

    HOThreshold = cms.double(1.1), ## GeV, Scheme B

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

    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0),

    # Method for momentum reconstruction
    MomConstrMethod = cms.int32(0),
    #Depth, fraction of the respective calorimeter [0,1]
    MomEmDepth = cms.double(0),
    MomHadDepth = cms.double(0),
    MomTotDepth = cms.double(0)
                          

)


