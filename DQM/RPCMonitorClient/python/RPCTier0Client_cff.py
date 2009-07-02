import FWCore.ParameterSet.Config as cms


# report summary
from DQM.RPCMonitorClient.RPCEventSummary_cfi import *
rpcEventSummary.Tier0 = True


rpcTier0Client = cms.Sequence(rpcEventSummary)
