import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("L1TMuonEmulation", eras.Run2_2016)
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.source = cms.Source('PoolSource',
 fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gflouris/public/SingleMuPt6180_noanti_10k_eta1.root')
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
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.load('L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi')

#process.load('L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi')
process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi')

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
    tauTokens = cms.untracked.VInputTag("simCaloStage2Digis"),
    jetToken = cms.untracked.InputTag("simCaloStage2Digis"),
    muonToken = cms.untracked.InputTag("simGmtStage2Digis"),
    sumToken = cms.untracked.InputTag("simCaloStage2Digis",""),
    maxL1Upgrade = cms.uint32(60)
)

process.L1TSeq = cms.Sequence(   process.SimL1Emulator
#                                   + process.dumpED
#                                   + process.dumpES
                                   + process.l1tSummaryStage2SimDigis
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
