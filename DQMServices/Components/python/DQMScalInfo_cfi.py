import FWCore.ParameterSet.Config as cms

dqmscalInfo = DQMStep1Module('DQMScalInfo',
    dqmScalFolder = cms.untracked.string('Scal')
)
