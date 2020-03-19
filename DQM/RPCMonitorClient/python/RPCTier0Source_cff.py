import FWCore.ParameterSet.Config as cms

from DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi import rpcdigidqm
rpcdigidqm.UseMuon =  cms.untracked.bool(True)

from DQM.RPCMonitorDigi.RPCRecHitProbability_cfi import rpcrechitprobability

# FED integrity
from DQM.RPCMonitorClient.RPCFEDIntegrity_cfi import rpcFEDIntegrity
from DQM.RPCMonitorClient.RPCMonitorRaw_cfi import *
from DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi import *


# DQM Services
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcEventInfo = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('RPC')
)

# DCS
from DQM.RPCMonitorDigi.RPCDcsInfo_cfi import *


rpcTier0Source = cms.Sequence(rpcdigidqm*rpcrechitprobability*rpcDcsInfo*rpcEventInfo*rpcFEDIntegrity)

