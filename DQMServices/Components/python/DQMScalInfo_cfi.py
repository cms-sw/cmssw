import FWCore.ParameterSet.Config as cms

dqmscalInfo = cms.EDAnalyzer("DQMScalInfo",
    dqmScalFolder = cms.untracked.string('Scal')
)
