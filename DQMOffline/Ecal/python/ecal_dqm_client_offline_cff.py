import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorClient.EcalMonitorClient_cfi import *

# placeholder
from DQM.EcalBarrelMonitorClient.EcalBarrelMonitorClient_cfi import *
from DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi import *

from DQMOffline.Ecal.EcalZmassClient_cfi import *

from DQM.EcalCommon.EcalDQMBinningService_cfi import *

# needs to be edited
dqmQTestEB = cms.EDAnalyzer("QualityTester",
#    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalBarrelMonitorModule/test/data/EcalBarrelQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndRun = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(False),
    verboseQT = cms.untracked.bool(False)
)

dqmQTestEE = cms.EDAnalyzer("QualityTester",
#    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalEndcapMonitorModule/test/data/EcalEndcapQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndRun = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(False),
    verboseQT = cms.untracked.bool(False)
)

ecal_dqm_client_offline = cms.Sequence(
    ecalMonitorClient *
    ecalzmassclient
)

ecalMonitorClient.clients = cms.untracked.vstring(
    "IntegrityClient",
    "OccupancyClient",
    "PresampleClient",
    "RawDataClient",
    "TimingClient",
    "SummaryClient"
)
    
