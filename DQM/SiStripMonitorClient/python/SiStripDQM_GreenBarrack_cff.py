# The following comments couldn't be translated into the new config version:

#--------------------------
# Web Service
#--------------------------

import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM services
#--------------------------
from DQMServices.Core.DQM_cfg import *
#  DQM Online Environment #####
# use include file for dqmEnv dqmSaver
from DQMServices.Components.DQMEnvironment_cfi import *
#--------------------------
# STRIP DQM Source and Client
#--------------------------
from DQM.SiStripMonitorClient.SiStripSourceConfig_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_cff import *
# Quality Tester ####
qTester = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

ModuleWebRegistry = cms.Service("ModuleWebRegistry")

SiStripDQMOnSimData = cms.Sequence(SiStripSourcesSimData*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOnRealData = cms.Sequence(SiStripSourcesRealDataTIF*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOffRealDataTIF = cms.Sequence(SiStripSourcesRealDataTIF*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOffSimData = cms.Sequence(SiStripSourcesSimData*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOffSimDataTest = cms.Sequence(cms.SequencePlaceholder("SiStripMonitorDigiSim")*cms.SequencePlaceholder("SiStripMonitorCluster")*cms.SequencePlaceholder("QualityMon")*cms.SequencePlaceholder("SiStripMonitorTrack")*cms.SequencePlaceholder("MonitorTrackResiduals")*cms.SequencePlaceholder("TrackMon")*qTester*SiStripOfflineDQMClient*dqmEnv*dqmSaver)
DQMStore.referenceFileName = 'Reference.root'
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
dqmSaver.saveByLumiSection = -1
# replace dqmSaver.saveByMinute = -1
# replace dqmSaver.saveByEvent =  -1
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = True

