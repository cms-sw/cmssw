import FWCore.ParameterSet.Config as cms


rpcMonitorRaw = cms.EDAnalyzer("RPCMonitorRaw",
  watchedErrors = cms.untracked.vint32(8,9),
  rpcRawDataCountsTag = cms.InputTag('rpcunpacker')
)
