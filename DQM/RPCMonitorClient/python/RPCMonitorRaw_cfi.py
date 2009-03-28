import FWCore.ParameterSet.Config as cms


rpcMonitorRaw = cms.EDAnalyzer("RPCMonitorRaw",
  writeHistograms = cms.untracked.bool(False),
  histoFileName = cms.untracked.string('rpcMonitorRaw.root')
)
