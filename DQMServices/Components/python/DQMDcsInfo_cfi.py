import FWCore.ParameterSet.Config as cms


dqmDcsInfo = cms.EDAnalyzer("DQMDcsInfo",
    subSystemFolder = cms.untracked.string('Info'),
    dcsInfoFolder = cms.untracked.string('DcsInfo')
)
