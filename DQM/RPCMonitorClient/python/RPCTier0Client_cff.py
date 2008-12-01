import FWCore.ParameterSet.Config as cms

#Daq info
from DQM.RPCMonitorClient.RPCFEDIntegrity_cfi import rpcDaqInfo


# report summary
from DQM.RPCMonitorClient.RPCEventSummary_cfi import *
rpcEventSummary.Tier0 = True


rpcTier0Client = cms.Sequence(rpcEventSummary * rpcDaqInfo)
