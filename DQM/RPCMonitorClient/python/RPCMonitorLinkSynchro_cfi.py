import FWCore.ParameterSet.Config as cms

rpcMonitorLinkSynchro = cms.EDAnalyzer("RPCMonitorLinkSynchro",
  dumpDelays = cms.untracked.bool(False)
)
