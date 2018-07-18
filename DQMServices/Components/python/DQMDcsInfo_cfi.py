import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmDcsInfo = DQMEDAnalyzer('DQMDcsInfo',
    subSystemFolder = cms.untracked.string('Info'),
    dcsInfoFolder = cms.untracked.string('DcsInfo')
)
