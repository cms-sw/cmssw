import FWCore.ParameterSet.Config as cms


rpcRawDataCount = cms.EDAnalyzer("RPCMonitorRaw",
  writeHistograms = cms.untracked.bool(False),
  histoFileName = cms.untracked.string('rpcMonitorRaw.root')
)
