import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM services
#--------------------------
from DQMServices.Core.DQM_cfg import *
#
# BEGIN DQM Online Environment #######################
#
# PUT THE FOLLOWING INTO YOUR PATH FOR OPERATION AT P5
# REPLACE YourSubsystemName by the name of your source ###
# use include file for dqmEnv dqmSaver
from DQMServices.Components.DQMEnvironment_cfi import *
# Possible conventions are "Online", "Offline" and "RelVal".
# Default is "Offline"
dqmSaver.convention = 'Online'
# replace dqmSaver.workflow      = ""
dqmSaver.dirName = '.'
# This is the filename prefix
dqmSaver.producer = 'DQM'
# (this goes into the foldername)
dqmEnv.subSystemFolder = 'SiStrip'
# Ignore run number for MC data
# replace dqmSaver.forceRunNumber  = -1
# optionally change fileSaving  conditions
# replace dqmSaver.saveByLumiSection =   1
# replace dqmSaver.saveByMinute = -1
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = True
# will add switch to select histograms to be saved soon
#
# END ################################################
#
# FIX YOUR  PATH TO INCLUDE dqmEnv and dqmSaver
#--------------------------
# STRIP DQM Source and Client
#--------------------------
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_Cosmic_cff import *
SiStripDQMTest_cosmicTk = cms.Sequence(SiStripDQMTier0_cosmicTk*dqmEnv*dqmSaver)
SiStripDQMTest_ckf = cms.Sequence(SiStripDQMTier0_ckf*dqmEnv*dqmSaver)
#SiStripDQMTest_rs = cms.Sequence(SiStripDQMTier0_rs*dqmEnv*dqmSaver)
SiStripDQMTest = cms.Sequence(SiStripDQMTier0*dqmEnv*dqmSaver)

