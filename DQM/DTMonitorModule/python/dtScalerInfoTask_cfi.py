import FWCore.ParameterSet.Config as cms

dtScalerInfoMonitor = cms.EDAnalyzer("DTScalerInfoTask",
    inputTagScaler = cms.untracked.InputTag('scalersRawToDigi'),
)


