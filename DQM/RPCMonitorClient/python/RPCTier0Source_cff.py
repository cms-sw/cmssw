import FWCore.ParameterSet.Config as cms

from DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi import *
rpcdigidqm.DigiEventsInterval = 100
rpcdigidqm.dqmshifter = True
rpcdigidqm.dqmexpert = True
rpcdigidqm.dqmsuperexpert = False
rpcdigidqm.DigiDQMSaveRootFile = False


# FED integrity
from DQM.RPCMonitorClient.RPCFEDIntegrity_cfi import rpcFEDIntegrity

# DQM Services
rpcEventInfo = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('RPC')
)


rpcTier0Source = cms.Sequence(rpcdigidqm*rpcEventInfo*rpcFEDIntegrity)
