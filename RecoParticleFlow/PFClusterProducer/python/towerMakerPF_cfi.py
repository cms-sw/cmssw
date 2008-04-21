# The following comments couldn't be translated into the new config version:

# GeV, Scheme B
# GeV, Scheme B
import FWCore.ParameterSet.Config as cms

towerMakerPF = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(-1000.0),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(True),
    EESumThreshold = cms.double(-1000.0),
    HOThreshold = cms.double(999999.0), ## GeV, Scheme B

    HBThreshold = cms.double(0.0), ## GeV, Scheme B

    EBThreshold = cms.double(999999.0), ## GeV, ORCA value w/o selective readout

    HcalThreshold = cms.double(-1000.0), ## GeV, -1000 means cut not used 

    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(False),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(999999.0), ## GeV, ORCA value

    EEThreshold = cms.double(999999.0), ## GeV, ORCA value w/o selective readout

    HESThreshold = cms.double(0.0), ## GeV, Scheme B

    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(999999.0), ## GeV, ORCA value

    HEDThreshold = cms.double(0.0), ## GeV, Scheme B

    EcutTower = cms.double(-1000.0), ## GeV, -1000 means cut not used

    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)


