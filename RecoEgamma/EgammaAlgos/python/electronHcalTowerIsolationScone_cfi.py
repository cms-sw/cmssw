import FWCore.ParameterSet.Config as cms

electronHcalTowerIsolationScone = cms.EDProducer("EgammaTowerIsolationProducer",
    absolut = cms.bool(True),
    intRadius = cms.double(0.15), # to be orthogonal with the H/E ID cut
    extRadius = cms.double(0.3),
    towerProducer = cms.InputTag("towerMaker"),
    etMin = cms.double(0.0),
    Depth = cms.int32(-1),
    emObjectProducer = cms.InputTag("gedGsfElectrons")
)


electronHcalDepth1TowerIsolationScone = cms.EDProducer("EgammaTowerIsolationProducer",
    absolut = cms.bool(True),
    intRadius = cms.double(0.15), # to be orthogonal with the H/E ID cut
    extRadius = cms.double(0.3),
    towerProducer = cms.InputTag("towerMaker"),
    etMin = cms.double(0.0),
    Depth = cms.int32(1),
    emObjectProducer = cms.InputTag("gedGsfElectrons")
)

electronHcalDepth2TowerIsolationScone = cms.EDProducer("EgammaTowerIsolationProducer",
    absolut = cms.bool(True),
    intRadius = cms.double(0.15), # to be orthogonal with the H/E ID cut
    extRadius = cms.double(0.3),
    towerProducer = cms.InputTag("towerMaker"),
    etMin = cms.double(0.0),
    Depth = cms.int32(2),
    emObjectProducer = cms.InputTag("gedGsfElectrons")
)

