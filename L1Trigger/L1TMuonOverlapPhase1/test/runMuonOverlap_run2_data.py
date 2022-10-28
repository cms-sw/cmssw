# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
        # suppressInfo       = cms.untracked.vstring('AfterSource', 'PostModule'),
        destinations=cms.untracked.vstring(
                                               # 'detailedInfo',
                                               # 'critical',
                                               #'cout',
                                               #'cerr',
                                                'omtfEventPrint'
                    ),
        categories=cms.untracked.vstring('l1tOmtfEventPrint', 'OMTFReconstruction'), #, 'FwkReport'
        # cout=cms.untracked.PSet(
        #                  threshold=cms.untracked.string('INFO'),
        #                  default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
        #                  # INFO   =  cms.untracked.int32(0),
        #                  # DEBUG   = cms.untracked.int32(0),
        #                  l1tOmtfEventPrint=cms.untracked.PSet(limit=cms.untracked.int32(1000000000)),
        #                  OMTFReconstruction=cms.untracked.PSet(limit=cms.untracked.int32(1000000000)),
        #                  #FwkReport=cms.untracked.PSet(reportEvery = cms.untracked.int32(50) ),
        #                ), 
        
        omtfEventPrint = cms.untracked.PSet(    
                         filename  = cms.untracked.string('log_MuonOverlap_run2_data'),
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('INFO'),
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         l1tOmtfEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) ),
                         OMTFReconstruction = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) )
                       ),
       debugModules=cms.untracked.vstring('simOmtfDigis') 
       # debugModules = cms.untracked.vstring('*')
    )

#process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False),
                                         # SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )

process.source = cms.Source('PoolSource',
 #fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gflouris/public/SingleMuPt6180_noanti_10k_eta1.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_15_p_1_1_qtl.root')    
 #fileNames = cms.untracked.vstring('file:///eos/user/k/kbunkow/cms_data/mc/PhaseIIFall17D/SingleMu_PU200_32DF01CC-A342-E811-9FE7-48D539F3863E_dump500Events.root')
# fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/mc/PhaseIITDRSpring19DR/PhaseIITDRSpring19DR_Mu_FlatPt2to100_noPU_v31_E0D5C6A5-B855-D14F-9124-0B2C9B28D0EA_dump4000Ev.root")
 fileNames = cms.untracked.vstring(
     #'/store/express/Commissioning2021/ExpressCosmics/FEVT/Express-v1/000/342/094/00000/038c179a-d2ce-45f0-a7d5-8b2d40017042.root',
     #'/store/express/Commissioning2021/ExpressCosmics/FEVT/Express-v1/000/344/566/00000/19ef107a-4cd9-4df0-ba93-dbfbab8df1cb.root'
     'file:///eos/user/k/kbunkow/cms_data/run2_data/Run2018D_ZeroBias_Run_325117_8BAB433D-F822-A64A-BB22-25E18AD5442F_allEv.root'
     ),                    
 )
	                    
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
#process.load('EventFilter.L1TRawToDigi.omtfStage2Digis_cfi') #unpacker

#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '113X_dataRun3_Prompt_v3', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '102X_upgrade2018_realistic_v16', '')
 


####Event Setup Producer
#process.load('L1Trigger.L1TMuonOverlapPhase1.fakeOmtfParams_cff')
#process.omtfParams.configXMLFile =  cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0006.xml")

process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
               data = cms.vstring('L1TMuonOverlapParams'))
                   ),
   verbose = cms.untracked.bool(False)
)

#process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis1.root'), closeFileFast = cms.untracked.bool(True) )
						
                        
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('EventFilter.L1TRawToDigi.omtfStage2Digis_cfi') #unpacker

		
####OMTF Emulator
process.load('L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi')

process.simOmtfDigis.srcDTPh = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcDTTh = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcCSC = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcRPC = cms.InputTag('omtfStage2Digis')

process.simOmtfDigis.bxMin = cms.int32(0)
process.simOmtfDigis.bxMax = cms.int32(0)

process.simOmtfDigis.dumpResultToXML = cms.bool(False)
process.simOmtfDigis.dumpResultToROOT = cms.bool(False)
process.simOmtfDigis.eventCaptureDebug = cms.bool(True)


#!!!!!!!!!!!!!!!!!!!!! all possible algorithm configuration parameters, if it is commented, then a defoult value is used
#below is the configuration used for runnig from the autumn of the 2018

#process.simOmtfDigis.sorterType = cms.string("byLLH")
# process.simOmtfDigis.ghostBusterType = cms.string("GhostBusterPreferRefDt")
#
# process.simOmtfDigis.minDtPhiQuality = cms.int32(2)
# process.simOmtfDigis.minDtPhiBQuality = cms.int32(2)
#
# process.simOmtfDigis.rpcMaxClusterSize = cms.int32(3)
# process.simOmtfDigis.rpcMaxClusterCnt = cms.int32(2)
# process.simOmtfDigis.rpcDropAllClustersIfMoreThanMax = cms.bool(False)
#
# process.simOmtfDigis.goldenPatternResultFinalizeFunction = cms.int32(0) #valid values are 0, 1, 2, 3, 5, 6, but for other then 0 the candidates quality assignemnt must be updated
#
# process.simOmtfDigis.noHitValueInPdf = cms.bool(False)

process.simOmtfDigis.lctCentralBx = cms.int32(8);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6




process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence(  process.esProd    +      
                                    process.omtfStage2Digis + process.simOmtfDigis 
                                   #+ process.dumpED
                                   #+ process.dumpES
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.out = cms.OutputModule("PoolOutputModule", 
   fileName = cms.untracked.string("l1tomtf_superprimitives1.root")
)

#process.output_step = cms.EndPath(process.out)
#process.schedule = cms.Schedule(process.L1TMuonPath)
#process.schedule.extend([process.output_step])
