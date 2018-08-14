import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcMonitorRaw = DQMEDAnalyzer('RPCMonitorRaw',
  watchedErrors = cms.untracked.vint32(8,9),
  rpcRawDataCountsTag = cms.InputTag('rpcunpacker')
)
