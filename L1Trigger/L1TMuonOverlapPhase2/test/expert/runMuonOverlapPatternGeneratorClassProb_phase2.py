# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
import re
from os import listdir
from os.path import isfile, join

process.load("FWCore.MessageLogger.MessageLogger_cfi")

verbose = True
versionIn = "ExtraplMB1andMB2RFixedP_ValueP1Scale_DT_2_2_2_t35_mcWaw2023_OneOverPt_and_iPt2"

#versionIn = "ExtraplMB1nadMB2DTQualAndEtaFixedP_ValueP1Scale_t20_SingleMu_mcWaw2023_OneOverPt"
#versionIn = "ExtraplMB1nadMB2DTQualAndEtaValueP1Scale_t18"
#versionIn = "0x00011_oldSample_3_30Files"

#versionOut =  versionIn + "_classProb17_recalib2" #_classProb17_recalib2_minDP0

versionOut =  "ExtraplMB1andMB2RFixedP_ValueP1Scale_DT_2_2_2_t35_" + "_classProb17_recalib2"

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
                         filename  = cms.untracked.string('Patterns_' + versionOut),
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
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(-1)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )
    
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.RawToDigi_cff')
#process.load('Configuration.StandardSequences.SimL1Emulator_cff')
#process.load('Configuration.StandardSequences.SimPhase2L1GlobalTriggerEmulator_cff')
#process.load('L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.prototypeSeeds')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '131X_mcRun4_realistic_v9', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '') 

path = '/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/' #old sample, but very big
#path = '/eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v2/' #new sample, but small and more noisy
#path = '/eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v1/'

#path = '/afs/cern.ch/work/a/akalinow/public/MuCorrelator/Data/SingleMu/9_3_14_FullEta_v1/'
#path = '/afs/cern.ch/work/k/kbunkow/public/data/SingleMuFullEta/721_FullEta_v4/'

chosenFiles = []

#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_p_10_' in f) or ('_m_10_' in f))]
chosenFiles.append('file://' + path + "SingleMu_18_p_1_1_rD8.root")


print("chosenFiles")
for chFile in chosenFiles:
    print(chFile)

if len(chosenFiles) == 0 :
    print("no files selected!!!!!!!!!!!!!!!")
    exit

firstEv = 0#40000
#nEvents = 1000

# input files (up to 255 files accepted)
process.source = cms.Source('PoolSource',
fileNames = cms.untracked.vstring( 
    #'file:/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_16_p_1_1_xTE.root',
    #'file:/afs/cern.ch/user/k/kpijanow/Neutrino_Pt-2to20_gun_50.root',
    list(chosenFiles), ),
    skipEvents =  cms.untracked.uint32(0),
    inputCommands=cms.untracked.vstring(
        'keep *',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
        'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016s_simEmtfDigis__HLT')
)
	                    
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

#process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis1_1.root'), closeFileFast = cms.untracked.bool(True) )
                                   
####OMTF Emulator
#process.load('L1Trigger.L1TMuonOverlapPhase1.simOmtfPhase2Digis_cfi')
process.load('L1Trigger.L1TMuonOverlapPhase2.simOmtfPhase2Digis_cfi')

process.simOmtfPhase2Digis.bxMin = cms.int32(0)
process.simOmtfPhase2Digis.bxMax = cms.int32(0)

process.simOmtfPhase2Digis.dumpResultToXML = cms.bool(False)
process.simOmtfPhase2Digis.eventCaptureDebug = cms.bool(False)

process.simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_template.xml")
#process.simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_0x00012_oldSample_3_30Files_grouped1_classProb1_recalib.xml")
#process.simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_0x00012_oldSample_3_30Files_grouped1_classProb11_recalib2.xml")
#process.simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0009_oldSample_3_10Files_classProb2.xml")
#process.simOmtfPhase2Digis.patternsXMLFiles = cms.VPSet(cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_plus_v1.xml")),
#                                                       cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_minus_v1.xml"))  )

#process.simOmtfPhase2Digis.patternsXMLFiles = cms.VPSet(#cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/PatternsDisplaced_0x0007_minus.xml")),
#                                                        cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/PatternsDisplaced_0x0007_plus.xml"))  
#                                                        )


#process.simOmtfPhase2Digis.patternGenerator = cms.string("modifyClassProb")
#process.simOmtfPhase2Digis.patternGenerator = cms.string("groupPatterns")
process.simOmtfPhase2Digis.patternGenerator = cms.string("patternGenFromStat")
#process.simOmtfPhase2Digis.patternGenerator = cms.string("") #does nothing except storing the patterns in the root file
#process.simOmtfPhase2Digis.patternsROOTFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_0x00011_oldSample_3_30Files_layerStat.root")
#process.simOmtfPhase2Digis.patternsROOTFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_layerStat_ExtraplMB1nadMB2_t14.root")
#process.simOmtfPhase2Digis.patternsROOTFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_layerStat_ExtraplMB1nadMB2FullAlgo_t16.root")
process.simOmtfPhase2Digis.patternsROOTFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_layerStat_" + versionIn + ".root")

process.simOmtfPhase2Digis.patternType = cms.string("GoldenPatternWithStat")
process.simOmtfPhase2Digis.generatePatterns = cms.bool(True)
#process.simOmtfPhase2Digis.optimisedPatsXmlFile = cms.string("Patterns_0x0009_oldSample_3_10Files_classProb3.xml")
#process.simOmtfPhase2Digis.optimisedPatsXmlFile = cms.string("Patterns_0x00012_oldSample_3_30Files_grouped1_classProb17_recalib2.xml")
#process.simOmtfPhase2Digis.optimisedPatsXmlFile = cms.string("Patterns_ExtraplMB1nadMB2FullAlgo_t16_classProb17_recalib2.xml")
process.simOmtfPhase2Digis.optimisedPatsXmlFile = cms.string("Patterns_" + versionOut + ".xml")
#process.simOmtfPhase2Digis.optimisedPatsXmlFile = cms.string("PatternsDisplaced_0x0007_p.xml")

process.simOmtfPhase2Digis.rpcMaxClusterSize = cms.int32(3)
process.simOmtfPhase2Digis.rpcMaxClusterCnt = cms.int32(2)
process.simOmtfPhase2Digis.rpcDropAllClustersIfMoreThanMax = cms.bool(True)

process.simOmtfPhase2Digis.goldenPatternResultFinalizeFunction = cms.int32(3) #valid values are 0, 1, 2, 3, 5
process.simOmtfPhase2Digis.lctCentralBx = cms.int32(6);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6

process.simOmtfPhase2Digis.simTracksTag = cms.InputTag('g4SimHits')

#process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
#process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence( process.simOmtfPhase2Digis 
                                   #+ process.dumpED
                                   #+ process.dumpES
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

#process.out = cms.OutputModule("PoolOutputModule", 
#   fileName = cms.untracked.string("l1tomtf_superprimitives1.root")
#)

#process.output_step = cms.EndPath(process.out)
#process.schedule = cms.Schedule(process.L1TMuonPath)
#process.schedule.extend([process.output_step])
