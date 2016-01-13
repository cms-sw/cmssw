import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras



#
# Use this Era to run the 2015 (Stage-1) Emulation
#
#process = cms.Process("L1TMuonEmulation", eras.Run2_25ns)
#
# Use this Era to run the 2016 (Stage-2) Emulation
#
process = cms.Process("L1TMuonEmulation", eras.Run2_2016)

import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))


process.source = cms.Source(
    'PoolSource',
# fileNames = cms.untracked.vstring('/store/data/Run2015B/DoubleEG/RAW/v1/000/251/251/00000/069C1D5D-EA25-E511-8377-02163E013807.root', 
#        '/store/data/Run2015B/DoubleEG/RAW/v1/000/251/251/00000/0A005856-EA25-E511-B409-02163E013542.root', 
#        '/store/data/Run2015B/DoubleEG/RAW/v1/000/251/251/00000/1ABA9855-EA25-E511-9141-02163E011A74.root', 
#        '/store/data/Run2015B/DoubleEG/RAW/v1/000/251/251/00000/F852D556-EA25-E511-ABF7-02163E011C17.root'),
 fileNames = cms.untracked.vstring('/store/data/Run2015B/DoubleMuon/RAW/v1/000/251/251/00000/9C66B64C-CA25-E511-818A-02163E0140E1.root', 
        '/store/data/Run2015B/DoubleMuon/RAW/v1/000/251/251/00000/DA60E454-CA25-E511-BFF0-02163E01207C.root'),
    lumisToProcess = cms.untracked.VLuminosityBlockRange("251251:1-251251:31", "251251:33-251251:97", "251251:99-251251:167"),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_hlt*_*_*',
        'drop *_sim*_*_*'
        ) 
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag

# Note: We are re-emulating L1T based on the conditions in the GT, so best for
#        now to use MC GT, even when running over a data file, and just ignore
#        the main DT TP emulator warnings...  (In future we'll override only L1T
#        emulator related conditions, so you can use a data GT)
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '') ## READ ABOVE BEFORE UNCOMMENTING 

#### Sim L1 Emulator Sequence:
process.load('Configuration.StandardSequences.RawToDigi_cff')
#process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('L1Trigger.Configuration.L1TReEmulateFromRAW_cff')
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.l1tSummaryA = cms.EDAnalyzer("L1TSummary")
process.l1tSummaryA.egCheck   = cms.bool(True);
process.l1tSummaryA.tauCheck  = cms.bool(True);
process.l1tSummaryA.jetCheck  = cms.bool(True);
process.l1tSummaryA.sumCheck  = cms.bool(True);
process.l1tSummaryA.muonCheck = cms.bool(True);
if (eras.stage1L1Trigger.isChosen()):
    process.l1tSummaryA.egToken   = cms.InputTag("caloStage1FinalDigis");
    process.l1tSummaryA.tauToken  = cms.InputTag("caloStage1FinalDigis:rlxTaus");
    process.l1tSummaryA.jetToken  = cms.InputTag("caloStage1FinalDigis");
    process.l1tSummaryA.sumToken  = cms.InputTag("caloStage1FinalDigis");
    process.l1tSummaryA.muonToken = cms.InputTag("None");
    process.l1tSummaryA.muonCheck = cms.bool(False);
if (eras.stage2L1Trigger.isChosen()):
    process.l1tSummaryA.egToken   = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.tauToken  = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.jetToken  = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.sumToken  = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.muonToken = cms.InputTag("gmtStage2Digis","");

process.l1tSummaryB = cms.EDAnalyzer("L1TSummary")
process.l1tSummaryB.egCheck   = cms.bool(True);
process.l1tSummaryB.tauCheck  = cms.bool(True);
process.l1tSummaryB.jetCheck  = cms.bool(True);
process.l1tSummaryB.sumCheck  = cms.bool(True);
process.l1tSummaryB.muonCheck = cms.bool(True);
if (eras.stage1L1Trigger.isChosen()):
    process.l1tSummaryB.egToken   = cms.InputTag("simCaloStage1FinalDigis");
    process.l1tSummaryB.tauToken  = cms.InputTag("simCaloStage1FinalDigis:rlxTaus");
    process.l1tSummaryB.jetToken  = cms.InputTag("simCaloStage1FinalDigis");
    process.l1tSummaryB.sumToken  = cms.InputTag("simCaloStage1FinalDigis");
    process.l1tSummaryB.muonToken = cms.InputTag("None");
    process.l1tSummaryB.muonCheck = cms.bool(False);
if (eras.stage2L1Trigger.isChosen()):
    process.l1tSummaryB.egToken   = cms.InputTag("simCaloStage2Digis");
    process.l1tSummaryB.tauToken  = cms.InputTag("simCaloStage2Digis");
    process.l1tSummaryB.jetToken  = cms.InputTag("simCaloStage2Digis");
    process.l1tSummaryB.sumToken  = cms.InputTag("simCaloStage2Digis");
    process.l1tSummaryB.muonToken = cms.InputTag("simGmtStage2Digis","");

# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t_debug.root')

# enable debug message logging for our modules
process.MessageLogger.categories.append('L1TCaloEvents')
process.MessageLogger.categories.append('L1TGlobalEvents')
process.MessageLogger.categories.append('l1t|Global')
process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

# gt analyzer
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
    doText = cms.untracked.bool(True),
    dmxEGToken = cms.InputTag("None"),
    dmxTauToken = cms.InputTag("None"),
    dmxJetToken = cms.InputTag("None"),
    dmxEtSumToken = cms.InputTag("None"),
    muToken = cms.InputTag("simGmtStage2Digis"),
    egToken = cms.InputTag("simCaloStage2Digis"),
    tauToken = cms.InputTag("simCaloStage2Digis"),
    jetToken = cms.InputTag("simCaloStage2Digis"),
    etSumToken = cms.InputTag("simCaloStage2Digis"),
    gtAlgToken = cms.InputTag("None"),
    emulDxAlgToken = cms.InputTag("simGlobalStage2Digis"),
    emulGtAlgToken = cms.InputTag("simGlobalStage2Digis")
)

# Stage-1 Ntuple will not contain muons, and might (investigating) miss some Taus.  (Work underway)
process.l1UpgradeTree = cms.EDAnalyzer("L1UpgradeTreeProducer")
if (eras.stage1L1Trigger.isChosen()):
    process.l1UpgradeTree.egToken      = cms.untracked.InputTag("simCaloStage1FinalDigis")
    process.l1UpgradeTree.tauToken     = cms.untracked.InputTag("simCaloStage1FinalDigis:rlxTaus")
    process.l1UpgradeTree.jetToken     = cms.untracked.InputTag("simCaloStage1FinalDigis")
    process.l1UpgradeTree.muonToken    = cms.untracked.InputTag("None")
    process.l1UpgradeTree.sumToken     = cms.untracked.InputTag("simCaloStage1FinalDigis","")
    process.l1UpgradeTree.maxL1Upgrade = cms.uint32(60)
if (eras.stage2L1Trigger.isChosen()):
    process.l1UpgradeTree.egToken      = cms.untracked.InputTag("simCaloStage2Digis")
    process.l1UpgradeTree.tauToken     = cms.untracked.InputTag("simCaloStage2Digis")
    process.l1UpgradeTree.jetToken     = cms.untracked.InputTag("simCaloStage2Digis")
    process.l1UpgradeTree.muonToken    = cms.untracked.InputTag("simGmtStage2Digis")
    process.l1UpgradeTree.sumToken     = cms.untracked.InputTag("simCaloStage2Digis","")
    process.l1UpgradeTree.maxL1Upgrade = cms.uint32(60)


if (eras.stage1L1Trigger.isChosen()):
    process.l1tSummaryA.egToken   = cms.InputTag("caloStage1FinalDigis");
    process.l1tSummaryA.tauToken  = cms.InputTag("caloStage1FinalDigis:rlxTaus");
    process.l1tSummaryA.jetToken  = cms.InputTag("caloStage1FinalDigis");
    process.l1tSummaryA.sumToken  = cms.InputTag("caloStage1FinalDigis");
    process.l1tSummaryA.muonToken = cms.InputTag("None");
    process.l1tSummaryA.muonCheck = cms.bool(False);
if (eras.stage2L1Trigger.isChosen()):
    process.l1tSummaryA.egToken   = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.tauToken  = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.jetToken  = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.sumToken  = cms.InputTag("caloStage2Digis");
    process.l1tSummaryA.muonToken = cms.InputTag("gmtStage2Digis","");


process.L1TSeq = cms.Sequence(   process.RawToDigi        
                                   + process.L1TReEmulateFromRAW
#                                   + process.dumpED
#                                   + process.dumpES
#                                   + process.l1tSummaryA
# Comment this next module to silence per event dump of L1T objects:
                                   + process.l1tSummaryB
#                                   + process.l1tGlobalAnalyzer
                                   + process.l1UpgradeTree
)

process.L1TPath = cms.Path(process.L1TSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1t.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TPath)
process.schedule.extend([process.output_step])

print process.L1TReEmulateFromRAW
