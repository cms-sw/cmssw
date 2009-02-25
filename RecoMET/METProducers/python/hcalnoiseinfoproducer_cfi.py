import FWCore.ParameterSet.Config as cms

hcalnoiseinfoproducer = cms.EDProducer(
    'HcalNoiseInfoProducer',
    fillDigis = cms.bool(True),
    fillRecHits = cms.bool(True),
    fillCaloTowers = cms.bool(True),
    HPDEnergyThreshold = cms.double(5.0),
    RBXEnergyThreshold = cms.double(5.0),
    recHitEnergyThreshold = cms.double(1.5),
    recHitTimeEnergyThreshold = cms.double(10.0),
    digiCollName = cms.string('hcalDigis'),
    recHitCollName = cms.string('hbhereco'),
    caloTowerCollName = cms.string('towerMaker'),
    requirePedestals = cms.bool(True),
    nominalPedestal = cms.double(2.5)
    )
