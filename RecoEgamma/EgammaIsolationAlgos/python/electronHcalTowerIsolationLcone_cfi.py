import FWCore.ParameterSet.Config as cms

electronHcalTowerIsolationLcone = cms.EDProducer("EgammaTowerIsolationProducer",
    absolut = cms.bool(True),
    intRadius = cms.double(0.15),  #to be orthogonal with H/E Egamma ID cut
    extRadius = cms.double(0.4),
    towerProducer = cms.InputTag("towerMaker"),
    etMin = cms.double(0.0),
    Depth = cms.int32(-1),
    emObjectProducer = cms.InputTag("gedGsfElectrons")
)


electronHcalDepth1TowerIsolationLcone = cms.EDProducer("EgammaTowerIsolationProducer",
    absolut = cms.bool(True),
    intRadius = cms.double(0.15),  #to be orthogonal with H/E Egamma ID cut
    extRadius = cms.double(0.4),
    towerProducer = cms.InputTag("towerMaker"),
    etMin = cms.double(0.0),
    Depth = cms.int32(1),
    emObjectProducer = cms.InputTag("gedGsfElectrons")
)

electronHcalDepth2TowerIsolationLcone = cms.EDProducer("EgammaTowerIsolationProducer",
    absolut = cms.bool(True),
    intRadius = cms.double(0.15),  #to be orthogonal with H/E Egamma ID cut
    extRadius = cms.double(0.4),
    towerProducer = cms.InputTag("towerMaker"),
    etMin = cms.double(0.0),
    Depth = cms.int32(2),
    emObjectProducer = cms.InputTag("gedGsfElectrons")
)
