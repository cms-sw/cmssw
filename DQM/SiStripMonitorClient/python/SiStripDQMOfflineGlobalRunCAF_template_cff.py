import FWCore.ParameterSet.Config as cms

# DQM SERVICES

# DQM Store
from DQMServices.Core.DQM_cfg import *
DQM.collectorHost = ''

#  DQM File Saving and Environment
from DQMServices.Components.DQMEnvironment_cfi import *
dqmSaver.convention   = 'Online'
dqmSaver.dirName      = 'OUTPUT_DIRECTORY'
dqmSaver.producer     = 'DQM'
dqmSaver.saveByRun    = 1
dqmSaver.saveAtJobEnd = True
dqmEnv.subSystemFolder = 'SiStrip'

# Quality Tester
qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor          = cms.untracked.int32(1),
    qtList                  = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

# DQM MODULES
from DQM.SiStripMonitorClient.SiStripDQMRecoConfigOfflineGlobalRunCAF_cfi   import *
from DQM.SiStripMonitorClient.SiStripDQMSourceConfigOfflineGlobalRunCAF_cfi import *
from DQM.SiStripMonitorClient.SiStripDQMClientConfigOfflineGlobalRunCAF_cfi import *
