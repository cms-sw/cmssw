import FWCore.ParameterSet.Config as cms

hihffilter = cms.EDProducer("HIhfFilter_miniAOD",
srcTowers = cms.InputTag("towerMaker")
)
