# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.source = cms.Source('PoolSource',
 fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gkaratha/private/bmtf/gen_samples/Singlemu_oneoverpt_100k.root')
# ,eventsToProcess=cms.untracked.VEventRange('1:1:969-1:1:969')

	                    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2400))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')



####Event Setup Producer
process.load('L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff')
process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonBarrelParamsRcd'),
               data = cms.vstring('L1TMuonBarrelParams'))
              ),
   verbose = cms.untracked.bool(True)
)


process.load('L1Trigger.L1TTwinMux.fakeTwinMuxParams_cff')
process.esProdTM = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TTwinMuxParamsRcd'),
               data = cms.vstring('L1TTwinMuxParams'))
              ),
   verbose = cms.untracked.bool(True)
)



####BMTF Emulator
process.load('L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi')
process.load('L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi')
process.simBmtfDigis.Debug = cms.untracked.int32(0)
process.simBmtfDigis.DTDigi_Source = cms.InputTag("simTwinMuxDigis")


###TwinMux Emulator
process.load('L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi')


process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence( process.esProd         
				   + process.esProdTM 
                                   + process.simTwinMuxDigis 
                                   + process.simBmtfDigis 
#                                   + process.dumpED
#                                   + process.dumpES
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 

    outputCommands = cms.untracked.vstring(
        'drop *',
        #'keep *CSC*_*_*_*',
        'keep *RPC*_*_*_*',
        'keep *DT*_*_*_*',
        'keep *L1Mu*_*_*_*',
        'keep *_*Muon*_*_*',
        'keep *_*gen*_*_*',
        'keep *_*TwinMux*_*_*',
        'keep *_*Bmtf*_*_*',
        'keep GenEventInfoProduct_generator_*_*'),	
   fileName = cms.untracked.string("l1tbmtf_emu_singlemu_debug.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])
