# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
#import commands
import re
from os import listdir
from os.path import isfile, join

process.load("FWCore.MessageLogger.MessageLogger_cfi")

verbose = True

filesNameLike = sys.argv[2]

#version = "ExtraplMB1nadMB2DTQualAndEtaFloatP_atan_ValueP1Scale_t18"
version = "noExtrapl_deltaPhiVsPhiRef"

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
                         filename  = cms.untracked.string("Patterns_dispalced_test_" + version + "_" + filesNameLike),
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('INFO'),
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         l1tOmtfEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) ),
                         OMTFReconstruction = cms.untracked.PSet( limit = cms.untracked.int32(1000000000) )
                       ),
       debugModules = cms.untracked.vstring('simOmtfDigis') 
       #debugModules = cms.untracked.vstring('*')
    )

    #process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
if not verbose:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(-1)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )
    
# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtendedRun4D86Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D86_cff')  
    
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtendedRun4D41Reco_cff')
#process.load('Configuration.Geometry.GeometryExtendedRun4D41_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '') 

#path = '/eos/user/c/cericeci/forOMTF/OMTF_Run3_FixedTiming/'
#path = '/eos/user/c/cericeci/forOMTF/OMTF_Run3_FixedTiming_FullOutput/'


#path = '/eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v2/' #new sample, but small and more noisy
#path = '/eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v1/'

#path = '/afs/cern.ch/work/a/akalinow/public/MuCorrelator/Data/SingleMu/9_3_14_FullEta_v1/'
#path = '/afs/cern.ch/work/k/kbunkow/public/data/SingleMuFullEta/721_FullEta_v4/'

#print(onlyfiles)

#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_p_10_' in f) or ('_m_10_' in f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_10_p_10_' in f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (re.match('.*_._p_10.*', f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if ((filesNameLike in f))]

#print(onlyfiles)

chosenFiles = []

cscBx = 8

if filesNameLike == 'displHighPt' : # displaced muon sample
    cscBx = 8
    #path = '/eos/user/c/cericeci/forOMTF/OMTF_PhaseII_FixedTiming/'
    path =  '/eos/user/a/asotorod/Samples/OMTF-L1/OMTF_fixedTiming/'
    
    fileCnt = 200 
    firstFile = 1 #1001            
    for i in range(firstFile, firstFile + fileCnt, 1):
        filePathName = path + "custom_Displaced_" + str(i) + "_numEvent5000.root"
        if isfile(filePathName) :
            #chosenFiles.append('file://' + path + "custom_Displaced_Run3_" + str(i) + "_numEvent1000.root") 
            #chosenFiles.append('file://' + path + "custom_Displaced_Run3_" + str(i) + "_numEvent2000.root") 
            chosenFiles.append('file://' + filePathName)
    
    print("chosenFiles")
    for chFile in chosenFiles:
        print(chFile)
    
    if len(chosenFiles) == 0 :
        print("no files selected!!!!!!!!!!!!!!! (argumetn should be e.g. 20_p")
        exit

elif filesNameLike == 'allPt' : # promt muon sample
    cscBx = 6
    path = '/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/' #old sample, but very big
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    
    filesPerPtBin = 10 #TODO max is 200 for the 721_FullEta_v4 and 100 for 9_3_14_FullEta_v2
    
    for ptCode in range(31, 4, -1) : #the rigt bound of range is not included 
        if ptCode == 5 : #5 is 3-4 GeV (maybe 3-3.5 GeV). 4 is 2-3GeV (maybe 2.5-3 GeV), very small fraction makes candidates, and even less reaches the second station
            filesPerPtBin = 30
        elif ptCode == 6 : #5 is 3-4 GeV (maybe 3-3.5 GeV). 4 is 2-3GeV (maybe 2.5-3 GeV), very small fraction makes candidates, and even less reaches the second station
            filesPerPtBin = 20    
        elif ptCode <= 7 : 
            filesPerPtBin = 10
        elif ptCode <= 12 :
            filesPerPtBin = 5
        else :    
            filesPerPtBin = 3
            
        filesPerPtBin = 1 # TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
            
        for sign in ['_m', '_p'] : #, m
            selFilesPerPtBin = 0
            for i in range(1, 50, 1): #TODO
                for f in onlyfiles:
                   if (( '_' + str(ptCode) + sign + '_' + str(i) + '_') in f): #TODO for 721_FullEta_v4/
                   #if (( '_' + str(ptCode) + sign + '_' + str(i) + ".") in f):  #TODO for 9_3_14_FullEta_v2
                        #print(f)
                        chosenFiles.append('file://' + path + f) 
                        selFilesPerPtBin += 1
                if(selFilesPerPtBin >= filesPerPtBin):
                    break

elif filesNameLike == 'mcWaw2022' :
    cscBx = 8
    path = '/eos/user/k/kbunkow/cms_data/mc/mcWaw2022/'
    chosenFiles.append('file://' + path + "DoubleMuPt1to100Eta24_1kevents.root") 
                            
else :
    cscBx = 6
    path = '/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/' #old sample, but very big
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    
    for i in range(1, 10, 1):
        for f in onlyfiles:
            if (( filesNameLike + '_' + str(i) + '_') in f):  #TODO for 721_FullEta_v4/
            #if (( filesNameLike + '_' + str(i) + '.') in f): #TODO for 9_3_14_FullEta_v2
                print(f)
                chosenFiles.append('file://' + path + f) 


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
process.omtfParams.configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0009_patGen.xml")

process.omtfParams.patternsXMLFiles = cms.VPSet(
        #cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")),
        #cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0009_oldSample_3_10Files_classProb1.xml") ),
        cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_template.xml")),
        #cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00012_oldSample_3_30Files_grouped1_classProb17_recalib2.xml")),
        #cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_layerStat_ExtraplMB1nadMB2_t10_classProb17_recalib2.xml")),
    )



process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
               data = cms.vstring('L1TMuonOverlapParams'))
                   ),
   verbose = cms.untracked.bool(False)
)

#process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis1_1.root'), closeFileFast = cms.untracked.bool(True) )
                                   
####OMTF Emulator
process.load('L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi')

#needed by candidateSimMuonMatcher
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.simOmtfDigis.candidateSimMuonMatcher = cms.bool(True)
process.simOmtfDigis.simTracksTag = cms.InputTag('g4SimHits')
process.simOmtfDigis.simVertexesTag = cms.InputTag('g4SimHits')
process.simOmtfDigis.muonMatcherFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/muonMatcherHists_100files_smoothStdDev_withOvf.root")


process.simOmtfDigis.bxMin = cms.int32(0)
process.simOmtfDigis.bxMax = cms.int32(0)

process.simOmtfDigis.dumpResultToXML = cms.bool(False)
process.simOmtfDigis.eventCaptureDebug = cms.bool(False)

process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/Patterns_template.xml")
#process.simOmtfDigis.patternsXMLFiles = cms.VPSet(cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_plus_v1.xml")),
#                                                       cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_minus_v1.xml"))
#)

#process.simOmtfDigis.patternGenerator = cms.string("patternGen")
#process.simOmtfDigis.patternGenerator = cms.string("2DHists")
process.simOmtfDigis.patternGenerator = cms.string("deltaPhiVsPhiRef")


process.simOmtfDigis.patternType = cms.string("GoldenPatternWithStat")
process.simOmtfDigis.generatePatterns = cms.bool(True)
process.simOmtfDigis.optimisedPatsXmlFile = cms.string("Patterns_dispalced_test_" + version + "_" + filesNameLike + ".xml")

process.simOmtfDigis.rpcMaxClusterSize = cms.int32(3)
process.simOmtfDigis.rpcMaxClusterCnt = cms.int32(2)
process.simOmtfDigis.rpcDropAllClustersIfMoreThanMax = cms.bool(True)

process.simOmtfDigis.minCSCStubRME12 = cms.int32(410) #[cm]
process.simOmtfDigis.minCSCStubR = cms.int32(490) #[cm]

process.simOmtfDigis.minDtPhiQuality = cms.int32(2)
process.simOmtfDigis.minDtPhiBQuality = cms.int32(4)

process.simOmtfDigis.dtRefHitMinQuality =  cms.int32(4)

#process.simOmtfDigis.usePhiBExtrapolationFromMB1 = cms.bool(True)
#process.simOmtfDigis.usePhiBExtrapolationFromMB2 = cms.bool(True)
process.simOmtfDigis.usePhiBExtrapolationFromMB1 = cms.bool(False)
process.simOmtfDigis.usePhiBExtrapolationFromMB2 = cms.bool(False)

process.simOmtfDigis.useStubQualInExtr  = cms.bool(True)
process.simOmtfDigis.useEndcapStubsRInExtr  = cms.bool(True)
process.simOmtfDigis.useFloatingPointExtrapolation  = cms.bool(True)
#process.simOmtfDigis.extrapolFactorsFilename = cms.FileInPath("ExtrapolationFactors_DTQualAndEtaValueP1Scale.xml")
#process.simOmtfDigis.extrapolFactorsFilename = cms.FileInPath("ExtrapolationFactors_simple.xml")
process.simOmtfDigis.extrapolFactorsFilename = cms.FileInPath("")

process.simOmtfDigis.stubEtaEncoding = cms.string("valueP1Scale")  
#process.simOmtfDigis.stubEtaEncoding = cms.string("bits")   

process.simOmtfDigis.goldenPatternResultFinalizeFunction = cms.int32(3) ## is needed here , becasue it just counts the number of layers with a stub
process.simOmtfDigis.lctCentralBx = cms.int32(cscBx);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6



#process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
#process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

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
