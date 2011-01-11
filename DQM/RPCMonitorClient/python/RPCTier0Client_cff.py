import FWCore.ParameterSet.Config as cms


# report summary
from DQM.RPCMonitorClient.RPCEventSummary_cfi import *

from DQM.RPCMonitorClient.RPCDqmClient_cfi import *

from  DQM.RPCMonitorClient.RPCChamberQuality_cfi import *

from  DQM.RPCMonitorClient.RPCEfficiencySecondStep_cfi import *

from  DQM.RPCMonitorClient.RPCEfficiencyShiftHisto_cfi import *

qTesterRPC = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(20)
)


# DCS
from DQM.RPCMonitorClient.RPCDcsInfoClient_cfi import *

rpcTier0Client = cms.Sequence(qTesterRPC*rpcdqmclient*rpcChamberQuality*rpcDcsInfoClient*rpcEventSummary*rpcefficiencysecond*rpcEfficiencyShiftHisto)



