import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
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

####Event Setup Producers
process.load('L1Trigger.L1TMuonBarrel.fakeMuonBarrelParams_cfi')
process.load('L1Trigger.L1TMuonOverlap.fakeMuonOverlapParams_cfi')
process.load('L1Trigger.L1TMuonEndCap.fakeMuonEndCapParams_cfi')
process.load('L1Trigger.L1TMuon.fakeMuonGlobalParams_cfi')


#### Emulators
process.load('L1Trigger.L1TMuonBarrel.simMuonBarrelDigis_cfi')
process.load('L1Trigger.L1TMuonOverlap.simMuonOverlapDigis_cfi')
process.load('L1Trigger.L1TMuonEndCap.simMuonEndCapDigis_cfi')
process.load('L1Trigger.L1TMuon.simMuonDigis_cfi')
process.load('L1Trigger.L1TCalorimeter.caloStage2Layer1Digis_cfi')

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

#process.l1tSummary = cms.EDAnalyzer("L1TSummary")
#process.l1tSummary.egToken   = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.tauToken  = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.jetToken  = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.sumToken  = cms.InputTag("simCaloStage2Digis");
#process.l1tSummary.muonToken = cms.InputTag("simGmtDigis","");
##process.l1tSummary.muonToken = cms.InputTag("simGmtDigis","imdMuonsBMTF");

process.load('L1Trigger.L1TCalorimeter.caloStage2Layer1Digis_cfi')
process.caloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
process.caloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")

process.L1TMuonSeq = cms.Sequence(   process.caloStage2Layer1Digis
                                   + process.simTwinMuxDigis
                                   + process.simBmtfDigis 
                                   + process.simEmtfDigis 
                                   + process.simOmtfDigis 
                                   + process.simGmtCaloSumDigis
                                   + process.simGmtDigis
#                                   + process.dumpED
#                                   + process.dumpES
#                                   + process.l1tSummary
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1tbmtf_superprimitives1.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])
