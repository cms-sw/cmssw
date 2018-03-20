import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmscalInfo = DQMEDAnalyzer('DQMScalInfo',
    dqmScalFolder = cms.untracked.string('Scal')
)
