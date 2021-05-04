import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorClient.EcalMonitorClient_cfi import *
from DQM.EcalCommon.EcalMEFormatter_cfi import ecalMEFormatter

from DQMServices.Core.DQMQualityTester import DQMQualityTester
dqmQTestEcal = DQMQualityTester(
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
    ecalMEFormatter
)

ecalMonitorClient.workers.remove('TrigPrimClient')
