import FWCore.ParameterSet.Config as cms

rpcMonitorLinkSynchro = DQMStep1Module('RPCMonitorLinkSynchro',
  dumpDelays = cms.untracked.bool(False),
  useFirstHitOnly = cms.untracked.bool(False),
  rpcRawSynchroProdItemTag = cms.InputTag('rpcunpacker')
)
