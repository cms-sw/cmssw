import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtChamberEfficiencyClient_cfi import *
from DQMOffline.MuonDPG.dtTnPEfficiencyClient_cfi import *
from DQM.DTMonitorClient.dtSegmentAnalysisTest_cfi import *
segmentTest.normalizeHistoPlots = True
segmentTest.runOnline = False
#segmentTest.detailedAnalysis = True
from DQM.DTMonitorClient.dtOfflineSummaryClients_cfi import *
from DQM.DTMonitorClient.dtResolutionAnalysisTest_cfi import *
from DQM.DTMonitorClient.dtTriggerEfficiencyTest_cfi import *
from DQM.DTMonitorClient.dtBlockedROChannelsTest_cfi import *
from DQM.DTMonitorClient.dtRunConditionVarClient_cfi import *
blockedROChannelTest.offlineMode = True;
from DQM.DTMonitorClient.ALCARECODTCalibSynchDQMClient_cff import *


from DQMServices.Core.DQMQualityTester import DQMQualityTester
dtQualityTests = DQMQualityTester(
                         #reportThreshold = cms.untracked.string('red'),
                         prescaleFactor = cms.untracked.int32(1),
                         qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests.xml'),
                         getQualityTestsFromFile = cms.untracked.bool(True)
                         )


dtClients = cms.Sequence(segmentTest+
                         dtResolutionAnalysisTest+
                         dtChamberEfficiencyClient+
                         dtTnPEfficiencyClient+
                         triggerEffTest+
                         blockedROChannelTest+
                         dtRunConditionVarClient+
                         dtOfflineSummaryClients+
                         ALCARECODTCalibSynchDQMClient)


