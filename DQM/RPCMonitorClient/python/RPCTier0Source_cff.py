import FWCore.ParameterSet.Config as cms

from DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi import rpcdigidqm
rpcdigidqm.UseMuon =  cms.untracked.bool(True)

from DQM.RPCMonitorDigi.RPCRecHitProbability_cfi import rpcrechitprobability

# FED integrity
from DQM.RPCMonitorClient.RPCFEDIntegrity_cfi import rpcFEDIntegrity
from DQM.RPCMonitorClient.RPCMonitorRaw_cfi import *
from DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi import *

# Efficiency
from DQM.RPCMonitorDigi.RPCEfficiency_cfi import *

# DQM Services
rpcEventInfo = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('RPC')
)

# DCS
from DQM.RPCMonitorDigi.RPCDcsInfo_cfi import *


rpcTier0Source = cms.Sequence(rpcdigidqm*rpcrechitprobability*rpcDcsInfo*rpcefficiency*rpcEventInfo*rpcFEDIntegrity)

