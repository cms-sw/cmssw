import FWCore.ParameterSet.Config as cms

muonFlag = True

# report summary
from DQM.RPCMonitorClient.RPCEventSummary_cfi import *

from DQM.RPCMonitorClient.RPCDqmClient_cfi import *


from DQM.RPCMonitorClient.RPCRecHitProbabilityClient_cfi import *

from  DQM.RPCMonitorClient.RPCChamberQuality_cfi import *

                                   
from  DQM.RPCMonitorClient.RPCEfficiencySecondStep_cfi import *

from  DQM.RPCMonitorClient.RPCEfficiencyShiftHisto_cfi import *
from  DQM.RPCMonitorClient.RPCEfficiencyPerRingLayer_cfi import *


qTesterRPC = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(20)
)


# DCS
from DQM.RPCMonitorClient.RPCDcsInfoClient_cfi import *

if (muonFlag):
    rpcTier0Client = cms.Sequence(qTesterRPC*rpcdqmclient*rpcdqmMuonclient*rpcrechitprobabilityclient*rpcChamberQuality*rpcMuonChamberQuality*rpcDcsInfoClient*rpcefficiencysecond*rpcEventSummary*rpcEfficiencyShiftHisto*rpcEfficiencyPerRingLayer)
else:
    rpcTier0Client = cms.Sequence(qTesterRPC*rpcdqmclient*rpcChamberQuality*rpcDcsInfoClient*rpcefficiencysecond*rpcEventSummary*rpcEfficiencyShiftHisto*rpcEfficiencyPerRingLayer)

