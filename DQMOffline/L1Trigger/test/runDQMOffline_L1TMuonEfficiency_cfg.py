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
    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/B85A468F-CAC7-E711-B1B0-02163E013250.root',
    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/2429B42E-D8C7-E711-8550-FA163E9318A5.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/2E6A28C4-B0C7-E711-B648-FA163EB276A5.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/2EA0EE1C-D9C7-E711-9C0C-FA163ED6B5D2.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3078B5E0-CEC7-E711-985E-FA163E185B6E.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/30A3CBD3-E5C7-E711-BFAB-FA163ED729A9.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/30B18A02-D1C7-E711-B575-FA163E87AF2A.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/30BA768D-BAC7-E711-BF68-02163E014EFD.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/321C6C19-C6C7-E711-9015-02163E0165CD.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/325D8D7B-93C7-E711-B473-FA163EA51782.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/32DA1865-87C8-E711-A6EF-FA163E7CE5E6.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/34D05C35-E0C7-E711-B3A3-FA163EE9E324.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3611D753-86C8-E711-A50A-FA163E0A1D04.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/361FC3C2-DCC7-E711-A9E3-02163E015C7F.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3809982E-D8C7-E711-B043-FA163EE8D072.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3821BFFA-EFC7-E711-9135-FA163EF59340.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3ABC66B6-CCC7-E711-9DBB-FA163EC23186.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3AD452AD-DBC7-E711-A0E3-FA163E393C4E.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3C54BCA6-ECC7-E711-99A1-02163E013E06.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3CC5246F-86C8-E711-AC3F-FA163EE20E28.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3CD764B6-BBC7-E711-B2B2-FA163EC648D9.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/3E5F4928-EAC7-E711-A17C-FA163EE8D072.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/402E0BEC-D4C7-E711-9EE8-02163E012E2F.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/4039CF9D-DBC7-E711-B6DD-FA163E595669.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/404D3245-E0C7-E711-B6F7-FA163EBADFE0.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/407F9E7F-D5C7-E711-A1A3-02163E00B25B.root',
#    '/store/data/Run2017F/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/306/456/00000/44FE3822-D9C7-E711-A5DB-FA163EF9815F.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
process.load('DQMOffline.L1Trigger.L1TMuonDQMEfficiency_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("TrackingTools.Configuration.TrackingTools_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
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
