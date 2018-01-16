import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcMonitorLinkSynchro = DQMEDAnalyzer('RPCMonitorLinkSynchro',
  dumpDelays = cms.untracked.bool(False),
  useFirstHitOnly = cms.untracked.bool(False),
  rpcRawSynchroProdItemTag = cms.InputTag('rpcunpacker')
)
