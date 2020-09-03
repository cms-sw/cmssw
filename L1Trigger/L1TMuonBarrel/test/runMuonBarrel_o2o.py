# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulationO2O")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.source = cms.Source('PoolSource',
 fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/gflouris/public/l1tbmtf.root')
	                    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


####Event Setup Producer
process.load('L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff')
process.fakeBmtfParams.configFromXML = cms.bool(True)
process.fakeBmtfParams.hwXmlFile = cms.string('L1Trigger/L1TMuon/data/o2o/bmtf/BMTF_HW.xml')
process.fakeBmtfParams.topCfgXmlFile = cms.string('L1Trigger/L1TMuon/data/o2o/bmtf/bmtf_top_config_p5.xml')

process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonBarrelParamsRcd'),
               data = cms.vstring('L1TMuonBarrelParams'))
                   ),
   verbose = cms.untracked.bool(True)
)


####BMTF Emulator
process.load('L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi')
process.load('L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi')
process.simBmtfDigis.Debug = cms.untracked.int32(0)

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence( process.esProd          
#				   + process.simTwinMuxDigis 
                                   + process.simBmtfDigis 
#                                   + process.dumpED
#                                   + process.dumpES
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1tbmtf.root")
)

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])
