import FWCore.ParameterSet.Config as cms


dqmProvInfo = cms.EDFilter("DQMProvInfo",
    subSystemFolder = cms.untracked.string('Info'),
    provInfoFolder = cms.untracked.string('ProvInfo')
)
