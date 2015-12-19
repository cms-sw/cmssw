import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
import commands

from Configuration.StandardSequences.Eras import eras
process = cms.Process("L1TMuonEmulation", eras.Run2_2016)

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

####Event Setup Producers
process.load('L1Trigger.L1TMuon.hackConditions_cff')

#### Emulators
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi')
process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")
process.load('L1Trigger.L1TMuonBarrel.simTwinMuxDigis_cfi')
process.load('L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi')
process.load('L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi')
process.load('L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi')
process.load('L1Trigger.L1TMuon.simGmtCaloSumDigis_cfi')
process.load('L1Trigger.L1TMuon.simGmtStage2Digis_cfi')

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

#process.l1tSummary = cms.EDAnalyzer("L1TSummary")
#process.l1tSummary.egToken   = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.tauToken  = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.jetToken  = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.sumToken  = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.muonToken = cms.InputTag("simGmtStage2Digis","");
##process.l1tSummary.muonToken = cms.InputTag("simGmtStage2Digis","imdMuonsBMTF");

process.L1TMuonSeq = cms.Sequence(   process.simCaloStage2Layer1Digis
                                   + process.simTwinMuxDigis
                                   + process.simBmtfDigis 
                                   + process.simEmtfDigis 
                                   + process.simOmtfDigis 
                                   + process.simGmtCaloSumDigis
                                   + process.simGmtStage2Digis
#                                   + process.dumpED
#                                   + process.dumpES
#                                   + process.l1tSummary
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1tmuon.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])



