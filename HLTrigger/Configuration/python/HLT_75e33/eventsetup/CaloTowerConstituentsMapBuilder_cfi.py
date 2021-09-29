import FWCore.ParameterSet.Config as cms

CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapAuto = cms.untracked.bool(False),
    MapFile = cms.untracked.string(''),
    SkipHE = cms.untracked.bool(True),
    appendToDataLabel = cms.string('')
)
