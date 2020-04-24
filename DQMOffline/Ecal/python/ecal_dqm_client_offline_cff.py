import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorClient.EcalMonitorClient_cfi import *
from DQM.EcalCommon.EcalMEFormatter_cfi import ecalMEFormatter

from DQMOffline.Ecal.EcalZmassClient_cfi import *

dqmQTestEcal = cms.EDAnalyzer("QualityTester",
#    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalMonitorClient/data/EcalQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndRun = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(False),
    verboseQT = cms.untracked.bool(False)
)

ecal_dqm_client_offline = cms.Sequence(
    ecalMonitorClient +
    ecalzmassclient +
    ecalMEFormatter
)
