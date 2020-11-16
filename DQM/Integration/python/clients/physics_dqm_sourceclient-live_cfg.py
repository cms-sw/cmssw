from __future__ import print_function
# $Id: physics_dqm_sourceclient-live_cfg.py,v 1.11 2012/02/13 15:09:30 lilopera Exp $

import FWCore.ParameterSet.Config as cms

process = cms.Process("Physics")

#----------------------------
# Event Source
#-----------------------------

# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'Physics'
process.dqmSaver.tag = 'Physics'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'Physics'
process.dqmSaverPB.runNumber = options.runNumber

# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('DQM/Physics/qcdLowPtDQM_cfi')

#process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.p = cms.Path(
    process.hltTriggerTypeFilter *
    process.myRecoSeq1  *
    process.myRecoSeq2  *
#    process.dump *
    process.qcdLowPtDQM *
    process.dqmEnv *
    process.dqmSaver *
    process.dqmSaverPB
)

process.siPixelDigis.cpu.InputLabel = cms.InputTag("rawDataCollector")

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())

if (process.runType.getRunType() == process.runType.hi_run):
    process.siPixelDigis.cpu.InputLabel = cms.InputTag("rawDataRepacker")
