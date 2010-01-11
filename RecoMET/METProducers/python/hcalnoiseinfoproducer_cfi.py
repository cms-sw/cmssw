import FWCore.ParameterSet.Config as cms

hcalnoise = cms.EDProducer(
    'HcalNoiseInfoProducer',
    fillDigis = cms.bool(True),
    fillRecHits = cms.bool(True),
    fillCaloTowers = cms.bool(True),
    fillTracks = cms.bool(True),

    # conditions for writing RBX to the event
    RBXEnergyThreshold = cms.double(10.),
    minRecHitEnergy = cms.double(1.5),
    maxProblemRBXs  = cms.int32(8),
    writeAllRBXs = cms.bool(False),

    # parameters for calculating summary variables
    maxCaloTowerIEta = cms.int32(20),
    maxTrackEta = cms.double(2.0),
    minTrackPt = cms.double(1.0),
    
    digiCollName = cms.string('hcalDigis'),
    recHitCollName = cms.string('hbhereco'),
    caloTowerCollName = cms.string('towerMaker'),
    trackCollName = cms.string('generalTracks'),
    hcalNoiseRBXCollName = cms.string('hcalnoiseinfoproducer'),
    
    )
