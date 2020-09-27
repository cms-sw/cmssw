import FWCore.ParameterSet.Config as cms

hiHFfilters = cms.EDProducer("HiHFFilterProducer",
srcTowers = cms.InputTag("towerMaker")
)
