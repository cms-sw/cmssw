import FWCore.ParameterSet.Config as cms


# report summary
from DQM.RPCMonitorClient.RPCEventSummary_cfi import *
rpcEventSummary.Tier0 = False

from DQM.RPCMonitorClient.RPCDqmClient_cfi import *

from  DQM.RPCMonitorClient.RPCChamberQuality_cfi import *


qTesterRPC = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(10)
)

rpcTier0Client = cms.Sequence(qTesterRPC*rpcdqmclient*rpcChamberQuality*rpcEventSummary)
