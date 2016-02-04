import FWCore.ParameterSet.Config as cms


dqmDcsInfoClient = cms.EDAnalyzer("DQMDcsInfoClient",
    subSystemFolder = cms.untracked.string('Info'),
    dcsInfoFolder = cms.untracked.string('DcsInfo')
)
