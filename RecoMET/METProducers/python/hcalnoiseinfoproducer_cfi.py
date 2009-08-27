import FWCore.ParameterSet.Config as cms

hcalnoise = cms.EDProducer(
    'HcalNoiseInfoProducer',
    fillDigis = cms.bool(True),
    fillRecHits = cms.bool(True),
    fillCaloTowers = cms.bool(True),
    fillJets = cms.bool(True),
    fillTracks = cms.bool(False),    
    dropRefVectors = cms.bool(False),
    refillRefVectors = cms.bool(False),

    # conditions for writing RBX to the event
    RBXEnergyThreshold = cms.double(10.),
    minRecHitEnergy = cms.double(1.5),
    maxProblemRBXs  = cms.int32(8),

    # parameters for calculating summary variables
    maxJetEmFraction = cms.double(0.05),
    maxJetEta = cms.double(3.5),
    maxCaloTowerIEta = cms.int32(29),
    maxTrackEta = cms.double(3.0),
    minTrackPt = cms.double(1.0),
    
    digiCollName = cms.string('hcalDigis'),
    recHitCollName = cms.string('hbhereco'),
    caloTowerCollName = cms.string('towerMaker'),
    caloJetCollName = cms.string('iterativeCone5CaloJets'),
    trackCollName = cms.string('generalTracks'),
    hcalNoiseRBXCollName = cms.string('hcalnoiseinfoproducer'),
    
    requirePedestals = cms.bool(True),
    nominalPedestal = cms.double(2.5)
    )
