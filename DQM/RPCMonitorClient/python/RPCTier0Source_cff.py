import FWCore.ParameterSet.Config as cms

# FED integrity
from DQM.RPCMonitorClient.RPCFEDIntegrity_cfi import rpcFEDIntegrity


# DQM Services
rpcEventInfo = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('RPC')
)




rpcTier0Source = cms.Sequence(rpcEventInfo*rpcFEDIntegrity)
