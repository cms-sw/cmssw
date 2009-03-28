import FWCore.ParameterSet.Config as cms


# report summary
from DQM.RPCMonitorClient.RPCEventSummary_cfi import *
rpcEventSummary.Tier0 = False

from DQM.RPCMonitorClient.RPCDqmClient_cfi import *

from  DQM.RPCMonitorClient.RPCChamberQuality_cfi import *
###rpcChamberQuality = cms.EDAnalyzer("RPCChamberQuality")

rpcTier0Client = cms.Sequence(rpcdqmclient*rpcEventSummary*rpcChamberQuality)
