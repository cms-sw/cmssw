# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
#import commands
import re
from os import listdir
from os.path import isfile, join
import fnmatch

process.load("FWCore.MessageLogger.MessageLogger_cfi")

verbose = True

filesNameLike = sys.argv[2]

version = "ExtraplMB1nadMB2DTQualAndEtaFixedP_ValueP1Scale_t20_v1_SingleMu"
#version = "noExtrapl_ValueP1Scale_t18_qualConverted_min4_ipT1_deltaPhiVsPhiRef_fixedDTScale"

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
                         filename  = cms.untracked.string('Patterns_layerStat_' + version + "_" + filesNameLike),
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('INFO'),
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
    
# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2026D86Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D86_cff')  
    
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '') 

chosenFiles = []

cscBx = 8

if filesNameLike == 'mcWaw2023' :
    cscBx = 8
    
    file_cnt = 100000
    for iPt in [0, 1, 2] :
        for charge in [0, 2] : 
            path = "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_04_04_2023/SingleMu_ch" + str(charge) + "_iPt" + str(iPt) + "_12_5_2_p1_04_04_2023/12_5_2_p1_04_04_2023/"
            #path = '/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_04_04_2023/SingleMu_ch0_iPt1_12_5_2_p1_04_04_2023/12_5_2_p1_04_04_2023/230404_084317/0000/'
              
            root_files = []
            for root, dirs, files in os.walk(path):
                for file in fnmatch.filter(files, 'SingleMu*.root'):
                    root_files.append(os.path.join(root, file))
                   
            file_num = 0    
            for root_file in root_files :
                if isfile(root_file) :
                    chosenFiles.append('file://' + root_file)
                    file_num += 1
                else :
                    print("file not found!!!!!!!: " + root_file)   
                    
                if file_num >= file_cnt :
                    break  
            
                        

    paths = ["/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_22_02_2023/SingleMu_ch0_iPt0_12_5_2_p1_22_02_2023/",
             "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_22_02_2023/SingleMu_ch2_iPt0_12_5_2_p1_22_02_2023/",
             "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_15_02_2023/SingleMu_ch0_iPt0_12_5_2_p1_15_02_2023/",
             "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_15_02_2023/SingleMu_ch2_iPt0_12_5_2_p1_15_02_2023/"]
    
    for path in paths :
        root_files = []
        for root, dirs, files in os.walk(path):
            for file in fnmatch.filter(files, 'SingleMu*.root'):
                root_files.append(os.path.join(root, file))  
                
        file_num = 0    
        for root_file in root_files :
            if isfile(root_file) :
                chosenFiles.append('file://' + root_file)
                file_num += 1
            else :
                print("file not found!!!!!!!: " + root_file)   
                
            if file_num >= file_cnt :
                break       

if filesNameLike == 'mcWaw2023_OneOverPt' :
    cscBx = 8
    
    file_cnt = 100000  
    paths = ["/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_20_04_2023/SingleMu_ch0_OneOverPt_12_5_2_p1_20_04_2023/",
             "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_20_04_2023/SingleMu_ch2_OneOverPt_12_5_2_p1_20_04_2023/",
             "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_14_04_2023/SingleMu_ch0_OneOverPt_12_5_2_p1_14_04_2023/",
             "/eos/user/a/akalinow/Data/SingleMu/12_5_2_p1_14_04_2023/SingleMu_ch2_OneOverPt_12_5_2_p1_14_04_2023/"]
    
    for path in paths :
        root_files = []
        for root, dirs, files in os.walk(path):
            for file in fnmatch.filter(files, 'SingleMu*.root'):
                root_files.append(os.path.join(root, file))  
                
        file_num = 0    
        for root_file in root_files :
            if isfile(root_file) :
                chosenFiles.append('file://' + root_file)
                file_num += 1
            else :
                print("file not found!!!!!!!: " + root_file)   
                
            if file_num >= file_cnt :
                break                 
    
print("chosenFiles")
for chFile in chosenFiles:
    print(chFile)

print("number of chosen files:", len(chosenFiles))

if len(chosenFiles) == 0 :
    print("no files selected!!!!!!!!!!!!!!! (argumetn should be e.g. 20_p")
    exit 
                           
# input files (up to 255 files accepted)
process.source = cms.Source('PoolSource',
fileNames = cms.untracked.vstring( 
    *(list(chosenFiles)) ),
    skipEvents =  cms.untracked.uint32(0),
    inputCommands=cms.untracked.vstring(
        'keep *',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
        'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016s_simEmtfDigis__HLT')
)
                    
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))


#Calibrate Digis
process.load("L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi")
process.CalibratedDigis.dtDigiTag = "simMuonDTDigis" 
process.CalibratedDigis.scenario = 0

#DTTriggerPhase2
process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.dtTriggerPhase2PrimitiveDigis.debug = False
process.dtTriggerPhase2PrimitiveDigis.dump = False
process.dtTriggerPhase2PrimitiveDigis.scenario = 0

####Event Setup Producer
process.load('L1Trigger.L1TMuonOverlapPhase1.fakeOmtfParams_cff')
process.omtfParams.configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0209_patGen.xml")
process.omtfParams.patternsXMLFiles = cms.VPSet(
        cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_template.xml")), )

process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
               data = cms.vstring('L1TMuonOverlapParams'))
                   ),
   verbose = cms.untracked.bool(False)
)

#process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis1_1.root'), closeFileFast = cms.untracked.bool(True) )
                                   
####OMTF Emulator
process.load('L1Trigger.L1TMuonOverlapPhase2.simOmtfPhase2Digis_extrapol_cfi')

#needed by candidateSimMuonMatcher
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

#process.simOmtfPhase2Digis.candidateSimMuonMatcher = cms.bool(True)
process.simOmtfPhase2Digis.simTracksTag = cms.InputTag('g4SimHits')
#process.simOmtfPhase2Digis.simVertexesTag = cms.InputTag('g4SimHits')
#process.simOmtfPhase2Digis.muonMatcherFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/muonMatcherHists_100files_smoothStdDev_withOvf.root")


process.simOmtfPhase2Digis.bxMin = cms.int32(0)
process.simOmtfPhase2Digis.bxMax = cms.int32(0)

process.simOmtfPhase2Digis.dumpResultToXML = cms.bool(False)
process.simOmtfPhase2Digis.eventCaptureDebug = cms.bool(False)

process.simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_template.xml")
#process.simOmtfPhase2Digis.patternsXMLFiles = cms.VPSet(cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_plus_v1.xml")),
#                                                       cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_minus_v1.xml"))
#)

process.simOmtfPhase2Digis.patternGenerator = cms.string("patternGen")

process.simOmtfPhase2Digis.patternType = cms.string("GoldenPatternWithStat")
process.simOmtfPhase2Digis.generatePatterns = cms.bool(True)
process.simOmtfPhase2Digis.cleanStubs = cms.bool(True) #has sense for the samples with secondaries

process.simOmtfPhase2Digis.optimisedPatsXmlFile = cms.string("Patterns_layerStat_" + version + "_" + filesNameLike + ".xml")

process.simOmtfPhase2Digis.rpcMaxClusterSize = cms.int32(3)
process.simOmtfPhase2Digis.rpcMaxClusterCnt = cms.int32(2)
process.simOmtfPhase2Digis.rpcDropAllClustersIfMoreThanMax = cms.bool(True)

process.simOmtfPhase2Digis.minCSCStubRME12 = cms.int32(410) #[cm]
process.simOmtfPhase2Digis.minCSCStubR = cms.int32(500) #[cm]

process.simOmtfPhase2Digis.minDtPhiQuality = cms.int32(2)
process.simOmtfPhase2Digis.minDtPhiBQuality = cms.int32(4)

process.simOmtfPhase2Digis.dtRefHitMinQuality =  cms.int32(4)

process.simOmtfPhase2Digis.usePhiBExtrapolationFromMB1 = cms.bool(True)
process.simOmtfPhase2Digis.usePhiBExtrapolationFromMB2 = cms.bool(True)

process.simOmtfPhase2Digis.useStubQualInExtr  = cms.bool(True)
process.simOmtfPhase2Digis.useEndcapStubsRInExtr  = cms.bool(True)
process.simOmtfPhase2Digis.useFloatingPointExtrapolation  = cms.bool(False)
#process.simOmtfPhase2Digis.extrapolFactorsFilename = cms.FileInPath("ExtrapolationFactors_ExtraplMB1nadMB2DTQualAndEta_ValueP1Scale_t20_SingleMu_iPt_100files_mcWaw2023.xml")

process.simOmtfPhase2Digis.stubEtaEncoding = cms.string("valueP1Scale")  

process.simOmtfPhase2Digis.goldenPatternResultFinalizeFunction = cms.int32(3) ## is needed here , becasue it just counts the number of layers with a stub

#process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
#process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence( process.esProd          
                                   + process.simOmtfPhase2Digis 
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
