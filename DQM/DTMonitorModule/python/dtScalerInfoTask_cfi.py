import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtScalerInfoMonitor = DQMEDAnalyzer('DTScalerInfoTask',
    inputTagScaler = cms.untracked.InputTag('scalersRawToDigi'),
)


