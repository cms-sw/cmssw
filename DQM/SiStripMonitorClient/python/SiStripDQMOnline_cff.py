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
from DQMServices.Components.test.dqm_onlineEnv_cfi import *
#--------------------------
# STRIP DQM Source and Client
#--------------------------
from DQM.SiStripMonitorClient.SiStripSourceConfig_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_cff import *
# Quality Tester ####
qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(200),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

ModuleWebRegistry = cms.Service("ModuleWebRegistry")

SiStripDQMOnSimData = cms.Sequence(SiStripSourcesSimData*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)
SiStripDQMOnRealData = cms.Sequence(SiStripSourcesRealData*qTester*SiStripOnlineDQMClient*dqmEnv*dqmSaver)
DQMStore.referenceFileName = 'Reference.root'
# put your subsystem name here: 
# DT, Ecal, Hcal, SiStrip, Pixel, RPC, CSC, L1T 
# (this goes into the filename)
dqmSaver.fileName = 'SiStrip'
dqmSaver.dirName = '.'
# # (this goes into the foldername)
dqmEnv.subSystemFolder = 'SiStrip'
#  DQM File Saving (optionally change fileSaving condition) #####
dqmSaver.prescaleLS = -1
dqmSaver.prescaleTime = -1 ## in minutes

dqmSaver.prescaleEvt = -1
dqmSaver.saveAtRunEnd = True
dqmSaver.saveAtJobEnd = False

