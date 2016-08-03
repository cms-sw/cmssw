# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TDQMOffline")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("DQMServices.Core.DQM_cfg")			
process.load('DQMOffline.Configuration.DQMOffline_cff')		
process.load('Configuration.EventContent.EventContent_cff')		


process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))
process.source = cms.Source('PoolSource',
	fileNames = cms.untracked.vstring(
	'file:///afs/cern.ch/work/a/astakia/public/dataset/3053A3BD-F844-E611-9A2B-02163E0138EC.root'			
	)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5))	


process.load('Configuration.StandardSequences.GeometryRecoDB_cff')   

process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')		

process.load("TrackingTools.Configuration.TrackingTools_cff")			


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_ICHEP16_repro_v0', '')		



process.load('DQMOffline.L1Trigger.L1TEfficiencyMuonsOffline_cff')

process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")		

process.l1tdumpeventsetup = cms.Path(process.dumpES)			

process.l1tEfficiencyMuons_offline.verbose   = cms.untracked.bool(False)

process.l1tEfficiencyMuons_offline.gmtInputTag  = cms.untracked.InputTag("gmtStage2Digis:Muon")		

process.L1TMuonSeq = cms.Sequence(
process.l1tEfficiencyMuons_offline          
)

process.L1TMuonPath = cms.Path(
	process.L1TMuonSeq
)

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('L1TOffline_L1TriggerOnly_job1_RAW2DIGI_RECO_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

process.output_step = cms.EndPath(process.DQMoutput)	

process.DQMoutput_step = cms.EndPath(process.DQMoutput)		

process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.dqmoffline_step = cms.Path(process.L1TMuonSeq)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.schedule = cms.Schedule(process.dqmoffline_step,process.endjob_step,process.DQMoutput_step)


