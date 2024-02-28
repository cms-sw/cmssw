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
#version = 't14_extrapolSimpl_displ_allfiles'
#version = 't16_extrapolSimpl_displ_test'
version = 'ExtraplMB1nadMB2FloatQualand_Eta_t17_v12_test_valueP1Scale'
#version = 'ExtraplMB1nadMB2SimplifiedFP_t17_v11_test_bits'
#version = 'Patterns_0x00012_t17_v11_extr_off_test_bits'

runDebug = "INFO" # or "INFO" DEBUG
useExtraploationAlgo = True
#useExtraploationAlgo = False

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
       categories        = cms.untracked.vstring( 'OMTFReconstruction', 'l1tOmtfEventPrint', 'l1MuonAnalyzerOmtf'), #'l1tOmtfEventPrint', 'l1MuonAnalyzerOmtf'
       omtfEventPrint = cms.untracked.PSet(    
                         filename  = cms.untracked.string('log_MuonOverlap_newPats_t' + version),
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string(runDebug), #DEBUG
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         l1tOmtfEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) ),
                         OMTFReconstruction = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) ),
                         l1MuonAnalyzerOmtf = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) ),
                       ),
       debugModules = cms.untracked.vstring('L1MuonAnalyzerOmtf', 'simOmtfDigis') #'L1MuonAnalyzerOmtf',
       #debugModules = cms.untracked.vstring('*')
    )

    #process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
if not verbose:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )
    
    
# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
#from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')    
    
    
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


#path = '/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/' #old sample, but very big
#path = '/eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v2/' #new sample, but small and more noisy
#path = '/eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v1/'


#path = '/afs/cern.ch/work/a/akalinow/public/MuCorrelator/Data/SingleMu/9_3_14_FullEta_v1/'
#path = '/afs/cern.ch/work/k/kbunkow/public/data/SingleMuFullEta/721_FullEta_v4/'

#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
#print(onlyfiles)

#filesNameLike = sys.argv[2]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_p_10_' in f) or ('_m_10_' in f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_10_p_10_' in f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (re.match('.*_._p_10.*', f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if ((filesNameLike in f))]

#print(onlyfiles)

chosenFiles = []

fileCnt = 1000 #1000 
if(runDebug == "DEBUG") :
    fileCnt = 1;
    
if True :    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #path = '/eos/user/c/cericeci/forOMTF/OMTF_PhaseII_FixedTiming/'    
    path =  '/eos/user/a/asotorod/Samples/OMTF-L1/OMTF_fixedTiming/'
    firstFile = 1 #1001            
    for i in range(firstFile, firstFile + fileCnt, 1):
        filePathName = path + "custom_Displaced_" + str(i) + "_numEvent5000.root"
        if isfile(filePathName) :
            chosenFiles.append('file://' + filePathName)
        else :
            print("file not found!!!!!!!: " + filePathName) 
# low pt
if False :
    path = '/eos/user/c/cericeci/forOMTF/OMTF_Run3_FixedTiming_FullOutput/'
    firstFile = 1001            
    for i in range(firstFile, firstFile + fileCnt, 1):
        #filePathName = path + "custom_Displaced_" + str(i) + "_numEvent5000.root"
        #chosenFiles.append('file://' + path + "custom_Displaced_Run3_" + str(i) + "_numEvent1000.root") 
        filePathName = path + "custom_Displaced_Run3_" + str(i) + "_numEvent2000.root" 
        if isfile(filePathName) :
            chosenFiles.append('file://' + filePathName)
        else :
            print("file not found!!!!!!!: " + filePathName)    


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
	                    
if(runDebug == "DEBUG") :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000))
else :
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))


####Event Setup Producer
process.load('L1Trigger.L1TMuonOverlapPhase1.fakeOmtfParams_cff')
process.omtfParams.configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0008.xml")

process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
               data = cms.vstring('L1TMuonOverlapParams'))
                   ),
   verbose = cms.untracked.bool(False)
)

analysisType = "efficiency" # or rate
  
for a in sys.argv :
    if a == "efficiency" or a ==  "rate" or a == "withTrackPart" :
        analysisType = a
        break;
    
print("analysisType=" + analysisType)

process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis2_eff_SingleMu_t' + version + '.root'), closeFileFast = cms.untracked.bool(True) )
                                   
####OMTF Emulator
if useExtraploationAlgo :
    process.load('L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_extrapolSimple_cfi')
else :
    process.load('L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi')    

if(runDebug == "DEBUG") :
    process.simOmtfDigis.dumpResultToXML = cms.bool(True)
    process.simOmtfDigis.XMLDumpFileName = cms.string("TestEvents_" + version + ".xml")
else :
    process.simOmtfDigis.dumpResultToXML = cms.bool(False)


if(runDebug == "DEBUG") :
    process.simOmtfDigis.eventCaptureDebug = cms.bool(True)
else :
    process.simOmtfDigis.eventCaptureDebug = cms.bool(False)    
#process.simOmtfDigis.simTracksTag = cms.InputTag('g4SimHits')

process.simOmtfDigis.sorterType = cms.string("byLLH")
process.simOmtfDigis.ghostBusterType = cms.string("byRefLayer") # byLLH byRefLayer GhostBusterPreferRefDt

if useExtraploationAlgo :
    #process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_layerStat_ExtraplMB1nadMB2_t10_classProb17_recalib2_test.xml")
    #process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_ExtraplMB1nadMB2Simplified_t14_classProb17_recalib2.xml")
    #process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_ExtraplMB1nadMB2FullAlgo_t16_classProb17_recalib2.xml")
    process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_ExtraplMB1nadMB2SimplifiedFP_t17_classProb17_recalib2.xml")
else :
    process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00012_oldSample_3_30Files_grouped1_classProb17_recalib2.xml")

  
process.simOmtfDigis.rpcMaxClusterSize = cms.int32(3)
process.simOmtfDigis.rpcMaxClusterCnt = cms.int32(2)
process.simOmtfDigis.rpcDropAllClustersIfMoreThanMax = cms.bool(True)

process.simOmtfDigis.minCSCStubRME12 = cms.int32(410) #[cm]
process.simOmtfDigis.minCSCStubR = cms.int32(490) #[cm]

process.simOmtfDigis.goldenPatternResultFinalizeFunction = cms.int32(10) #valid values are 0, 1, 2, 3, 5

process.simOmtfDigis.noHitValueInPdf = cms.bool(True)

process.simOmtfDigis.minDtPhiQuality = cms.int32(2)
process.simOmtfDigis.minDtPhiBQuality = cms.int32(4)

process.simOmtfDigis.lctCentralBx = cms.int32(8);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6

if useExtraploationAlgo :
    process.simOmtfDigis.dtRefHitMinQuality =  cms.int32(4)

    process.simOmtfDigis.usePhiBExtrapolationFromMB1 = cms.bool(True)
    process.simOmtfDigis.usePhiBExtrapolationFromMB2 = cms.bool(True)
    
    process.simOmtfDigis.useStubQualInExtr  = cms.bool(True)
    process.simOmtfDigis.useEndcapStubsRInExtr  = cms.bool(True)
    process.simOmtfDigis.useFloatingPointExtrapolation  = cms.bool(True)
    process.simOmtfDigis.extrapolFactorsFilename = cms.string("")
    
process.simOmtfDigis.stubEtaEncoding = cms.string("valueP1Scale")  
#process.simOmtfDigis.stubEtaEncoding = cms.string("bits")   

#process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
#process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")




process.L1TMuonSeq = cms.Sequence( process.esProd          
                                   + process.simOmtfDigis 
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
