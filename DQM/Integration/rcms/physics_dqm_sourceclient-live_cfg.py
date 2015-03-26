# $Id: physics_dqm_sourceclient-live_cfg.py,v 1.8 2010/02/10 11:13:54 lilopera Exp $

import FWCore.ParameterSet.Config as cms

process = cms.Process("Physics")

#----------------------------
# Event Source
#-----------------------------

process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'Physics DQM Consumer'
#process.EventStreamHttpReader.sourceURL = "http://localhost:50082/urn:xdaq-application:lid=29"

#filter on specific trigger types
#process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
#    SelectEvents = cms.vstring('HLT_Activity*','HLT_MinBias*','HLT_L1Tech_HCAL_HF_coincidence_PM','HLT_*BSC','HLT_HFThreshold*','HLT_L1*','HLT_*SingleTrack')
#) 

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'Physics'

# 0=random, 1=physics, 2=calibration, 3=technical
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
process.hltTriggerTypeFilter.SelectedTriggerType = 1

#---- for P5 (online) DB access
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

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
    process.dqmSaver
)
