import FWCore.ParameterSet.Config as cms

# Web Service
ModuleWebRegistry = cms.Service("ModuleWebRegistry")

# DQM services
from DQMServices.Core.DQM_cfg import *
DQMStore.referenceFileName = 'Reference.root'

#  DQM Online Environment #####
# use include file for dqmEnv dqmSaver
from DQMServices.Components.DQMEnvironment_cfi import *
dqmSaver.convention = 'Online'
dqmSaver.dirName    = "."
dqmSaver.producer   = 'DQM'
dqmEnv.subSystemFolder = 'SiStrip'
dqmSaver.saveByLumiSection = -1
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = True

# Quality Tester ####
from DQMServices.Core.DQMQualityTester import DQMQualityTester
qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

# STRIP DQM Source and Client ####
from DQM.SiStripMonitorClient.SiStripSourceConfig_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_cff import *


SiStripDQMOnSimData = cms.Sequence(SiStripSourcesSimData*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOnRealData = cms.Sequence(SiStripSourcesRealData*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)

SiStripDQMOnRealDataCollision = cms.Sequence(SiStripSourcesRealDataCollision*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)
