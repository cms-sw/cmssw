import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmProvInfo = DQMEDAnalyzer('DQMProvInfo',
    subSystemFolder = cms.untracked.string('Info'),
    provInfoFolder = cms.untracked.string('ProvInfo'),
    runType = cms.untracked.string("No Run Type selected")
)
