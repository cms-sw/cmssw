# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys

process.load("FWCore.MessageLogger.MessageLogger_cfi")

verbose = True

if verbose: 
    process.MessageLogger = cms.Service("MessageLogger",
       #suppressInfo       = cms.untracked.vstring('AfterSource', 'PostModule'),
       destinations   = cms.untracked.vstring(
                                               #'detailedInfo',
                                               #'critical',
                                               #'cout',
                                               #'cerr',
                                               'omtfEventPrint'
                    ),
       categories        = cms.untracked.vstring('l1tOmtfEventPrint', 'OMTFReconstruction'),
       omtfEventPrint = cms.untracked.PSet(    
                         filename  = cms.untracked.string('log_MuonOverlap_v0209'),
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('DEBUG'),
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         l1tOmtfEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) ),
                         OMTFReconstruction = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) )
                       ),
       debugModules = cms.untracked.vstring('simOmtfPhase2Digis') 
       #debugModules = cms.untracked.vstring('*')
    )

    #process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
if not verbose:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
										 #SkipEvent = cms.untracked.vstring('ProductNotFound') 
									 )                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 

process.source = cms.Source('PoolSource',  
  fileNames = cms.untracked.vstring('file:///eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/OMTF/13_1_0_03_04_2024/SingleMu_ch0_OneOverPt_Run2029_13_1_0_03_04_2024/13_1_0_03_04_2024/240403_080928/0000/SingleMu_OneOverPt_1_100_m_1.root')                  
 )
	                    
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D95Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.RawToDigi_cff')
#process.load('Configuration.StandardSequences.SimL1Emulator_cff')
#process.load('Configuration.StandardSequences.SimPhase2L1GlobalTriggerEmulator_cff')
#process.load('L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.prototypeSeeds')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '131X_mcRun4_realistic_v9', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '') 

process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis1.root'), closeFileFast = cms.untracked.bool(True) )
		
# Calibrate Digis
process.load("L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi")
process.CalibratedDigis.dtDigiTag = "simMuonDTDigis" 
process.CalibratedDigis.scenario = 0

# DTTriggerPhase2
process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.dtTriggerPhase2PrimitiveDigis.debug = False
process.dtTriggerPhase2PrimitiveDigis.dump = False
process.dtTriggerPhase2PrimitiveDigis.scenario = 0
	
								
####OMTF Emulator
process.load('L1Trigger.L1TMuonOverlapPhase2.simOmtfPhase2Digis_cfi')

process.simOmtfPhase2Digis.dumpResultToXML = cms.bool(True)
process.simOmtfPhase2Digis.eventCaptureDebug = cms.bool(True)

process.simOmtfPhase2Digis.lctCentralBx = cms.int32(8);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6

# process.simOmtfPhase2Digis.rpcMaxClusterSize = cms.int32(3)
# process.simOmtfPhase2Digis.rpcMaxClusterCnt = cms.int32(2)
# process.simOmtfPhase2Digis.rpcDropAllClustersIfMoreThanMax = cms.bool(False)
#
# process.simOmtfPhase2Digis.dropDTPrimitives = cms.bool(True)
# process.simOmtfPhase2Digis.usePhase2DTPrimitives = cms.bool(True)

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence( #process.esProd+        
                                   process.simOmtfPhase2Digis 
                                   #+ process.dumpED
                                   #+ process.dumpES
)

                            #TODO tune on dtTriggerPhase2PrimitiveDigis re-prodction if needed 
process.L1TMuonPath = cms.Path(#process.CalibratedDigis * process.dtTriggerPhase2PrimitiveDigis * 
                               process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1tomtf_superprimitives1.root")
)

#process.output_step = cms.EndPath(process.out)
#process.schedule = cms.Schedule(process.L1TMuonPath)
#process.schedule.extend([process.output_step])
