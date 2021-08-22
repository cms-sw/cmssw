# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
import commands

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
       categories        = cms.untracked.vstring('l1tOmtfEventPrint', 'L1T', 'OmtfUnpacker::produce', ''), #'OMTFReconstruction',
       omtfEventPrint = cms.untracked.PSet(    
                         filename  = cms.untracked.string('log_MuonOverlap_nn'),
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('DEBUG'),
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         l1tOmtfEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(10000000) ),
                         #OMTFReconstruction = cms.untracked.PSet( limit = cms.untracked.int32(10000000) )
                       ),
#        cout = cms.untracked.PSet(    
#                          threshold = cms.untracked.string('INFO'),
#                          default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
#                          #INFO   =  cms.untracked.int32(0),
#                          #DEBUG   = cms.untracked.int32(0),
#                          l1tOmtfEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(1000000) ),
#                          #OMTFReconstruction = cms.untracked.PSet( limit = cms.untracked.int32(1000000) )
#                        ),
       #debugModules = cms.untracked.vstring('L1MuonAnalyzerOmtfPhase1', 'simOmtfDigis') 
       debugModules = cms.untracked.vstring('omtfStage2Digis', 'L1MuonAnalyzerOmtfHW')  #, 'simOmtfDigis'
       
       #debugModules = cms.untracked.vstring('L1MuonAnalyzerOmtf', 'simOmtfDigis', 'omtfStage2Digis') 
       #debugModules = cms.untracked.vstring('*')
    )

    #process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
if not verbose:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('EventFilter.L1TRawToDigi.omtfStage2Digis_cfi') #unpacker

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')


process.source = cms.Source('PoolSource',
 #fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gflouris/public/SingleMuPt6180_noanti_10k_eta1.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_15_p_1_1_qtl.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_20_p_100_2_B7Z.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_5_p_1_1_Meh.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_7_p_1_1_DkC.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_18_p_1_1_2KD.root')
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_0_pre4/src/L1Trigger/L1TMuonOverlapPhase1/test/expert/DisplacedMuonGun_Pt30To100_Dxy_0_1000_E68C6334-7F62-E911-8AA5-0025905B8610_dump2000Ev.root')
 
 #fileNames = cms.untracked.vstring('file:///eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_31_p_101_2_DzU.root')
 #fileNames = cms.untracked.vstring('file:///eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v2/SingleMu_6_p_1.root'),
 #fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/mc/PhaseIITDRSpring19DR/PhaseIITDRSpring19DR_Mu_FlatPt2to100_noPU_v31_E0D5C6A5-B855-D14F-9124-0B2C9B28D0EA_dump4000Ev.root"),
 #fileNames = cms.untracked.vstring('file:///eos/home-k/konec/FFCFF986-ED0B-B74F-B253-C511D19B8249.root'),
 #fileNames = cms.untracked.vstring('file:///afs/cern.ch/user/k/konec/work/CMSSW_10_6_1_patch2.displaced/src/UserCode/OmtfAnalysis/jobs/FFCFF986-ED0B-B74F-B253-C511D19B8249.root'),
 
 #fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/run2_data/Run2018D_ZeroBias_CB56F74E-F55A-B247-AB06-D1A7406AB671_1000Ev.root"),
 #fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/run2_data/Run2018D_ZeroBias_501FAD58-6212-8F46-812C-759AF2603F81_allEv.root"),
 #fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/run2_data/Run2018D_ZeroBias_Run_325117_8BAB433D-F822-A64A-BB22-25E18AD5442F_allEv.root"),
 #fileNames = cms.untracked.vstring("/store/mc/Run3Winter20DRMiniAOD/Mu_FlatPt1to1000-pythia8-gun/GEN-SIM-RAW/NoPU_110X_mcRun3_2021_realistic_v6-v3/10000/003C515F-E4D1-404D-8921-36A3FD7361E9.root"),
 #fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/mc/mcRun3/Run3Winter20_Mu_FlatPt1to1000_mcRun3_2021_003C515F-E4D1-404D-8921-36A3FD7361E9_300Ev.root"),
 fileNames = cms.untracked.vstring("file:///eos/user/k/kbunkow/cms_data/mc/mcRun3/Run3Winter20_HTo2LongLivedTo4mu_MH-125_mcRun3_2021_03FD2A52-9B9A-544B-816F-8BF926F15CE8.root"),
 
        inputCommands=cms.untracked.vstring(
        'keep *',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
        'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016s_simEmtfDigis__HLT')
)
	                    
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))


####Event Setup Producer
############# L1TMuonOverlapPhase1
#process.load('L1Trigger.L1TMuonOverlapPhase1.fakeOmtfParams_cff')
#process.omtfParams.configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0008.xml")

# process.load('L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff')
# process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
#    toGet = cms.VPSet(
#       cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
#                data = cms.vstring('L1TMuonOverlapParams'))
#                    ),
#    verbose = cms.untracked.bool(False)
# )

process.TFileService = cms.Service("TFileService", fileName = cms.string('omtfAnalysis2.root'), closeFileFast = cms.untracked.bool(True) )
                

from EventFilter.L1TRawToDigi.omtfStage2Raw_cfi import *
process.load('EventFilter.L1TRawToDigi.omtfStage2Raw_cfi') #packer
                                   
#### new OMTF Emulator
process.load('L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi')

process.simOmtfDigis.bxMin = cms.int32(0)
process.simOmtfDigis.bxMax = cms.int32(0)

process.simOmtfDigis.srcDTPh = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcDTTh = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcCSC = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcRPC = cms.InputTag('omtfStage2Digis')
#process.simOmtfDigis.dropRPCPrimitives = cms.bool(False)
#process.simOmtfDigis.dropDTPrimitives = cms.bool(False)
#process.simOmtfDigis.dropCSCPrimitives = cms.bool(False)
process.simOmtfDigis.bxMin = cms.int32(0)
process.simOmtfDigis.bxMax = cms.int32(0)

process.simOmtfDigis.dumpResultToXML = cms.bool(True)
process.simOmtfDigis.dumpResultToROOT = cms.bool(False)
process.simOmtfDigis.eventCaptureDebug = cms.bool(False)

#process.simOmtfDigis.ghostBusterType = cms.string("GhostBusterPreferRefDt") #GhostBusterPreferRefDt byLLH

#Run3 config
#process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0009_oldSample_3_10Files.xml")
#process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0009_oldSample_3_10Files_classProb1.xml")
#process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00011_oldSample_3_30Files_grouped1_classProb7.xml")
#process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00012_oldSample_3_30Files_grouped1_classProb1_recalib.xml")

#process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")
#process.simOmtfDigis.patternsXMLFiles = cms.VPSet(cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_plus_v1.xml")),
#                                                       cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/GPs_parametrised_minus_v1.xml"))
#)
#process.simOmtfDigis.sorterType = cms.string("byLLH")


# process.simOmtfDigis.minDtPhiQuality = cms.int32(2)
# process.simOmtfDigis.minDtPhiBQuality = cms.int32(4)
#   
# process.simOmtfDigis.rpcMaxClusterSize = cms.int32(3)
# process.simOmtfDigis.rpcMaxClusterCnt = cms.int32(2)
# process.simOmtfDigis.rpcDropAllClustersIfMoreThanMax = cms.bool(True)
# 
# process.simOmtfDigis.goldenPatternResultFinalizeFunction = cms.int32(9) #valid values are 0, 1, 2, 3, 5
# 
# process.simOmtfDigis.noHitValueInPdf = cms.bool(True)
# 
# process.simOmtfDigis.lctCentralBx = cms.int32(8);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6

#Run2 config
#process.omtfParams.configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0006.xml")

process.simOmtfDigis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")
process.simOmtfDigis.minDtPhiQuality = cms.int32(2)
process.simOmtfDigis.minDtPhiBQuality = cms.int32(2)
  
process.simOmtfDigis.rpcMaxClusterSize = cms.int32(3)
process.simOmtfDigis.rpcMaxClusterCnt = cms.int32(2)
process.simOmtfDigis.rpcDropAllClustersIfMoreThanMax = cms.bool(False)

process.simOmtfDigis.goldenPatternResultFinalizeFunction = cms.int32(0) #valid values are 0, 1, 2, 3, 5

process.simOmtfDigis.noHitValueInPdf = cms.bool(False)

process.simOmtfDigis.lctCentralBx = cms.int32(8);#<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!TODO this was changed in CMSSW 10(?) to 8. if the data were generated with the previous CMSSW then you have to use 6



#nn_pThresholds = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54 ]
#nn_pThresholds = [0.40, 0.50] 
#nn_pThresholds = [0.35, 0.40, 0.45, 0.50, 0.55] 
 
#process.simOmtfDigis.neuralNetworkFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/omtfClassifier_withPtBins_v34.txt")
#process.simOmtfDigis.ptCalibrationFileName = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/PtCalibration_v34.root")

#process.simOmtfDigis.nn_pThresholds = cms.vdouble(nn_pThresholds)

# end of L1TMuonOverlapPhase1

#process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
#process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

analysisType = "rate" #efficiency or rate 
  
for a in sys.argv :
    if a == "efficiency" or a ==  "rate" or a == "withTrackPart" :
        analysisType = a
        break;
    
print "analysisType=" + analysisType

process.L1MuonAnalyzerOmtfPhase1 = cms.EDAnalyzer("L1MuonAnalyzerOmtf", 
                                 etaCutFrom = cms.double(0.82), #OMTF eta range
                                 etaCutTo = cms.double(1.24),
                                 L1OMTFInputTag  = cms.InputTag("simOmtfDigis","OMTF"),
                                 #nn_pThresholds = cms.vdouble(nn_pThresholds), 
                                 analysisType = cms.string(analysisType),
                                 
                                 #simTracksTag = cms.InputTag('g4SimHits'),
                                 #simVertexesTag = cms.InputTag('g4SimHits'),
                                 #trackingParticleTag = cms.InputTag("mix", "MergedTrackTruth"),
                                 #TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                 
                                 #matchUsingPropagation = cms.bool(True),
                                 #muonMatcherFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/muonMatcherHists_100files_smoothStdDev_withOvf.root") #if you want to make this file, remove this entry
                                 )


process.L1TMuonSeqPhase1 = cms.Sequence( process.omtfStage2Digis #+ process.esProd          
                                   + process.simOmtfDigis
                                   + process.L1MuonAnalyzerOmtfPhase1
                                   #+ process.dumpED
                                   #+ process.dumpES
)

process.L1TMuonPathPhase1 = cms.Path(process.L1TMuonSeqPhase1)



####OMTF Emulator
process.load('L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi')
process.simOmtfDigis.srcDTPh = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcDTTh = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcCSC = cms.InputTag('omtfStage2Digis')
process.simOmtfDigis.srcRPC = cms.InputTag('omtfStage2Digis')
# process.simOmtfDigis.dropRPCPrimitives = cms.bool(False)
# process.simOmtfDigis.dropDTPrimitives = cms.bool(False)
# process.simOmtfDigis.dropCSCPrimitives = cms.bool(False)
process.simOmtfDigis.bxMin = cms.int32(0)
process.simOmtfDigis.bxMax = cms.int32(0)
process.simOmtfDigis.dumpResultToXML = cms.bool(True)


process.L1MuonAnalyzerOmtfHW = cms.EDAnalyzer("L1MuonAnalyzerOmtf", 
                                 etaCutFrom = cms.double(0.82), #OMTF eta range
                                 etaCutTo = cms.double(1.24),
                                 L1OMTFInputTag  = cms.InputTag("omtfStage2Digis"), #candidates stored from hardware
                                 #L1OMTFInputTag  = cms.InputTag("simOmtfDigis", "OMTF"), #,"OMTF"
                                 #nn_pThresholds = cms.vdouble(nn_pThresholds), 
                                 analysisType = cms.string(analysisType),
                                 
                                 #simTracksTag = cms.InputTag('g4SimHits'),
                                 #simVertexesTag = cms.InputTag('g4SimHits'),
                                 #trackingParticleTag = cms.InputTag("mix", "MergedTrackTruth"),
                                 #TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                 
                                 #matchUsingPropagation = cms.bool(True),
                                 #muonMatcherFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/muonMatcherHists_100files_smoothStdDev_withOvf.root") #if you want to make this file, remove this entry
                                 )

process.L1TMuonSeq = cms.Sequence( process.simOmtfDigis)

process.L1TMuonPath = cms.Path( process.omtfStage2Digis + process.L1MuonAnalyzerOmtfHW) #process.omtfStage2Raw +  process.L1TMuonSeq + + process.L1MuonAnalyzerOmtfHW


#process.L1TMuonPathOmtfStage2 = cms.Path(process.omtfStage2Digis + process.esProd + process.simOmtfDigis + process.L1MuonAnalyzerOmtfHW) #process.simOmtfDigis + 

#process.schedule = cms.Schedule(process.L1TMuonPathPhase1)
#process.schedule = cms.Schedule(process.L1TMuonPathOmtfStage2) #process.L1TMuonPathPhase1,  process.L1TMuonPathEmulator)
process.schedule = cms.Schedule(process.L1TMuonPath)

#process.out = cms.OutputModule("PoolOutputModule", 
#   fileName = cms.untracked.string("l1tomtf_superprimitives1.root")
#)

#process.output_step = cms.EndPath(process.out)
#process.schedule = cms.Schedule(process.L1TMuonPath)
#process.schedule.extend([process.output_step])
