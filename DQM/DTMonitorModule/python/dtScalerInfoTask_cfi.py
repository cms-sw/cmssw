import FWCore.ParameterSet.Config as cms

dtScalerInfoMonitor = DQMStep1Module('DTScalerInfoTask',
    inputTagScaler = cms.untracked.InputTag('scalersRawToDigi'),
)


