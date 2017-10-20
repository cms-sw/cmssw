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

import FWCore.ParameterSet.Config as cms

# DQM file saver module
dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    # Possible conventions are "Online", "Offline" and "RelVal".
    convention = cms.untracked.string('Offline'),
    # Save files in plain ROOT or encode ROOT objects in ProtocolBuffer
    fileFormat = cms.untracked.string('ROOT'),
    # Name of the producer.
    producer = cms.untracked.string('DQM'),
    # Name of the processing workflow.
    workflow = cms.untracked.string(''),
    # Directory in which to save the files.
    dirName = cms.untracked.string('.'),
    # Only save this directory
    filterName = cms.untracked.string(''),
    # Version name to be used in file name.
    version = cms.untracked.int32(1),
    # runIsComplete
    runIsComplete = cms.untracked.bool(False),
    # Save file every N lumi sections (-1: disabled)
    saveByLumiSection = cms.untracked.int32(-1),
    # Save file every N runs (-1: disabled)
    saveByRun = cms.untracked.int32(-1),
    # Save file at the end of the job
    saveAtJobEnd = cms.untracked.bool(True),
    # Ignore run number for MC data (-1: disabled)
    forceRunNumber = cms.untracked.int32(-1),
    # Control reference saving (default / skip / qtests / all)
    referenceHandling = cms.untracked.string('all'),
    # Control which references are saved for qtests (default: STATUS_OK)
    referenceRequireStatus = cms.untracked.int32(100)
)

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
    '/store/data/Run2017B/SingleMuon/RAW-RECO/ZMu-PromptReco-v2/000/298/996/00000/4AAA21A6-186A-E711-A5B2-02163E019BD7.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
process.load('DQMOffline.L1Trigger.L1TMuonDQMEfficiency_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("TrackingTools.Configuration.TrackingTools_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_ICHEP16_repro_v0', '')
process.load('DQMOffline.L1Trigger.L1TMuonDQMOffline_cfi')
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
process.l1tdumpeventsetup = cms.Path(process.dumpES)
process.l1tMuonDQMOffline.verbose   = cms.untracked.bool(False)
process.l1tMuonDQMOffline.gmtInputTag  = cms.untracked.InputTag("gmtStage2Digis:Muon")
process.L1TMuonSeq = cms.Sequence(process.l1tMuonDQMOffline)
process.L1TMuonPath = cms.Path(process.L1TMuonSeq)
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/RelVal/DQMOffline/L1Trigger'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.ppost = cms.EndPath(process.l1tMuonDQMEfficiency + process.dqmSaver)
