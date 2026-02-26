import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorClient.EcalMonitorClient_cff import *
from DQM.EcalCommon.EcalMEFormatter_cfi import ecalMEFormatter

from DQMOffline.Ecal.EcalZmassClient_cfi import *

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
    ecalzmassclient +
    ecalMEFormatter
)

ecalMonitorClient.workerParameters.TrigPrimClient.params.sourceFromEmul = False

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toReplaceWith(ecalMonitorClient, ecalMonitorClientPhase2)
phase2_ecal_devel.toReplaceWith(ecal_dqm_client_offline, ecal_dqm_client_offline.copyAndExclude([ecalzmassclient]))
