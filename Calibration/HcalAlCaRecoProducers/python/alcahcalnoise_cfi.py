import FWCore.ParameterSet.Config as cms

# producer for alcaminbisas (HCAL minimum bias)

HcalNoiseProd = cms.EDProducer("AlCaEcalHcalReadoutsProducer",
    JetSource = cms.InputTag("iterativeCone5CaloJets"),
    MetSource = cms.InputTag("met"),
    TowerSource = cms.InputTag("towerMaker"),
    UseJet = cms.bool(True),
    UseMET = cms.bool(False),
    MetCut = cms.double(0),
    JetMinE = cms.double(20),
    JetHCALminEnergyFraction = cms.double(0.98),

    hbheInput = cms.InputTag("hbhereco"),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco"),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    ecalPSInput = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    rawInput = cms.InputTag("rawDataCollector")
)


