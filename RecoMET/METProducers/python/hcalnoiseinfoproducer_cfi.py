import FWCore.ParameterSet.Config as cms

hcalnoiseinfoproducer = cms.EDProducer(
    'HcalNoiseInfoProducer',
    fillDigis = cms.bool(True),
    fillRecHits = cms.bool(True),
    fillCaloTowers = cms.bool(True),
    fillJets = cms.bool(True),
    dropRefVectors = cms.bool(True),
    refillRefVectors = cms.bool(False),
    HPDEnergyThreshold = cms.double(50.),
    RBXEnergyThreshold = cms.double(50.),
    maxProblemRBXs = cms.int32(8),
    maxJetEmFraction = cms.double(0.05),
    digiCollName = cms.string('hcalDigis'),
    recHitCollName = cms.string('hbhereco'),
    caloTowerCollName = cms.string('towerMaker'),
    caloJetCollName = cms.string('iterativeCone5CaloJets'),
    hcalNoiseRBXCollName = cms.string('hcalnoiseinfoproducer'),
    requirePedestals = cms.bool(True),
    nominalPedestal = cms.double(2.5)
    )
