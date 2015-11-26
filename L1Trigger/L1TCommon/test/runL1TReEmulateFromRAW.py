import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("L1TMuonEmulation", eras.Run2_2016)
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/mulhearn/public/data/raw_76x.root'),
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

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

process.l1UpgradeTree = cms.EDAnalyzer(
    "L1UpgradeTreeProducer",
    egToken = cms.untracked.InputTag("simCaloStage2Digis"),
    tauToken = cms.untracked.InputTag("simCaloStage2Digis"),
    jetToken = cms.untracked.InputTag("simCaloStage2Digis"),
    muonToken = cms.untracked.InputTag("simGmtStage2Digis"),
    sumToken = cms.untracked.InputTag("simCaloStage2Digis",""),
    maxL1Upgrade = cms.uint32(60)
)

process.L1TSeq = cms.Sequence(   process.RawToDigi        
#                                   + process.SimL1Emulator
                                   + process.L1TReEmulateFromRAW
                                   + process.dumpED
#                                   + process.dumpES
                                   + process.l1tSummaryA
                                   + process.l1tSummaryB
#                                   + process.l1tGlobalAnalyzer
#                                   + process.l1UpgradeTree
)

process.L1TPath = cms.Path(process.L1TSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1t.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TPath)
process.schedule.extend([process.output_step])
