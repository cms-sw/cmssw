import FWCore.ParameterSet.Config as cms

# DQM services
DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

#  DQM Online Environment #####
# use include file for dqmEnv dqmSaver
from DQMServices.Components.DQMEnvironment_cfi import *
dqmSaver.convention = 'Online'
dqmSaver.dirName       = "."
dqmSaver.producer = 'DQM'
dqmEnv.subSystemFolder = 'SiStrip'
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = False

# Quality Tester ####
from DQMServices.Core.DQMQualityTester import DQMQualityTester
qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

# STRIP DQM Source and Client
from DQM.SiStripMonitorClient.SiStripSourceConfig_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_cff import *

SiStripDQMOffSimData = cms.Sequence(SiStripSourcesSimData*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOffRealData = cms.Sequence(SiStripSourcesRealData*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOffRealDataCollision = cms.Sequence(SiStripSourcesRealDataCollision*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
