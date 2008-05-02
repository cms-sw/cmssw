# The following comments couldn't be translated into the new config version:

#--------------------------
# DQM services
#--------------------------

import FWCore.ParameterSet.Config as cms

#  DQM Online Environment #####
# use include file for dqmEnv dqmSaver
from DQMServices.Components.DQMEnvironment_cfi import *
#--------------------------
# STRIP DQM Source and Client
#--------------------------
from DQM.SiStripMonitorClient.SiStripSourceConfig_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_cff import *
DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

# Quality Tester ####
qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(200),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

SiStripDQMOffSimData = cms.Sequence(SiStripSourcesSimData*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOffRealDataTIF = cms.Sequence(SiStripSourcesRealDataTIF*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
# Possible conventions are "Online", "Offline" and "RelVal".
# Default is "Offline"
dqmSaver.convention = 'Online'
#replace dqmSaver.workflow      = "/A/B/C"
# replace dqmSaver.dirName       = "."
# This is the filename prefix
dqmSaver.producer = 'DQM'
# (this goes into the foldername)
dqmEnv.subSystemFolder = 'SiStrip'
# Ignore run number for MC data
# replace dqmSaver.forceRunNumber  = -1
# optionally change fileSaving  conditions
# replace dqmSaver.saveByLumiSection =  -1
# replace dqmSaver.saveByMinute = -1
# replace dqmSaver.saveByEvent =  -1
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = False

