# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

verbose = False

if verbose: 
    process.MessageLogger = cms.Service("MessageLogger",
       #suppressInfo       = cms.untracked.vstring('AfterSource', 'PostModule'),
       destinations   = cms.untracked.vstring(
                                               #'detailedInfo',
                                               #'critical',
                                               #'cout',
                                               #'cerr',
                                               'omtfEventDump'
                    ),
       categories        = cms.untracked.vstring('MuTimingModule', 'omtfEventPrintout'), #, 
       omtfEventDump = cms.untracked.PSet(    
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('DEBUG'),
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         omtfEventPrintout = cms.untracked.PSet( limit = cms.untracked.int32(100000000) )
                       ),
       debugModules = cms.untracked.vstring('L1TMuonOverlapTTMergerTrackProducer', 'OmtfTTAnalyzer', 'simOmtfDigis', 'omtfTTAnalyzer', 'simBayesMuCorrelatorTrackProducer') 
       #debugModules = cms.untracked.vstring('*')
    )

    #process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
if not verbose:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )


#######################################TTTracks################################################
GEOMETRY = "D17"

process.load('Configuration.StandardSequences.Services_cff')
#process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

if GEOMETRY == "D17":
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
elif GEOMETRY == "TkOnly":
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('L1Trigger.TrackTrigger.TkOnlyTiltedGeom_cff')
else:
    print "this is not a valid geometry!!!"

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

fileNames = []
filesCnt = 100
path = 'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v5/'
for i in range(1, filesCnt+1, 1):
    fileNames.append(path + 'HSCPppstau_M_494_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_' + str(i) + '.root')
 
path = 'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v7/'
for i in range(1, filesCnt+1, 1):
    fileNames.append(path + 'HSCPppstau_M_1599_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_' + str(i) + '.root')

#path = 'file:///eos/user/a/akalinow/Data/SingleMu/9_3_14_FullEta_v2/'
#for i in range(1, filesCnt+1, 1):
#    if i != 44: #missing file
#        fileNames.append(path + 'SingleMu_20_m_' + str(i) + '.root')
#for i in range(1, filesCnt+1, 1):
#    fileNames.append(path + 'SingleMu_20_p_' + str(i) + '.root')
    
# Source_Files = cms.untracked.vstring(
#          'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v5/HSCPppstau_M_494_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_1.root',
#          'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v5/HSCPppstau_M_494_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_2.root',
#          'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v5/HSCPppstau_M_494_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_3.root',
#          
#          
#          'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v7/HSCPppstau_M_1599_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_1.root',
#          'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v7/HSCPppstau_M_1599_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_2.root',
#          'file:///eos/user/a/akalinow/Data/9_3_14_HSCP_v7/HSCPppstau_M_1599_TuneZ2star_13TeV_pythia6_cff_py_GEN_SIM_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_3.root',     
# )

Source_Files = cms.untracked.vstring(fileNames)


process.source = cms.Source("PoolSource", fileNames = Source_Files,
        inputCommands=cms.untracked.vstring(
        'keep *',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
        'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016s_simEmtfDigis__HLT')
)

process.TFileService = cms.Service("TFileService", fileName = cms.string('muCorrelatorTTAnalysis1HSCP.root'), closeFileFast = cms.untracked.bool(True))


############################################################
# remake L1 stubs and/or cluster/stub truth ??
############################################################

process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")

#if GEOMETRY == "D10": 
#    TTStubAlgorithm_official_Phase2TrackerDigi_.zMatchingPS = cms.bool(False)

if GEOMETRY != "TkOnly": 
    from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *
    TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")

process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
process.TTClusterStubTruth = cms.Path(process.TrackTriggerAssociatorClustersStubs)


############################################################
# L1 tracking
############################################################

#from L1Trigger.TrackFindingTracklet.Tracklet_cfi import *
#if GEOMETRY == "D10": 
#    TTTracksFromTracklet.trackerGeometry = cms.untracked.string("flat")
#TTTracksFromTracklet.asciiFileName = cms.untracked.string("evlist.txt")

process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
process.TTTracks = cms.Path(process.L1TrackletTracks)
process.TTTracksWithTruth = cms.Path(process.L1TrackletTracksWithAssociators)


#######################################TTTracks################################################

##This overrides the tracker geometry and the TTTriger does not work!!!!!!!!!!!!
# # PostLS1 geometry used
# process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2015_cff')
# ############################
# process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
# from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


####Event Setup Producer
# process.load('L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff')
# process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
#    toGet = cms.VPSet(
#       cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
#                data = cms.vstriL1TMuonBayesrlapParams'))
#                    ),
#    verbose = cms.untracked.bool(False)
# )


####OMTF Emulator
process.load('L1Trigger.L1TMuonBayes.simBayesMuCorrelatorTrackProducer_cfi')

#process.TFileService = cms.Service("TFileService", fileName = cms.string('muCorrelatorHists.root'), closeFileFast = cms.untracked.bool(True))

#process.simBayesMuCorrelatorTrackProducer.ttTracksSource = cms.string("L1_TRACKER")
#process.simBayesMuCorrelatorTrackProducer.ttTracksSource = cms.string("SIM_TRACKS") #TODO !!!!!!!
process.simBayesMuCorrelatorTrackProducer.ttTracksSource = cms.string("TRACKING_PARTICLES") #TODO !!!!!!!
process.simBayesMuCorrelatorTrackProducer.pdfModuleType = cms.string("PdfModuleWithStats") #TODO
#process.simBayesMuCorrelatorTrackProducer.pdfModuleFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/pdfModule.xml") #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!11
#process.simBayesMuCorrelatorTrackProducer.pdfModuleFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/pdfModule.xml")
process.simBayesMuCorrelatorTrackProducer.useStubsFromAdditionalBxs = cms.int32(3)

process.simBayesMuCorrelatorTrackProducer.generateTiming = cms.bool(False)
#process.simBayesMuCorrelatorTrackProducer.outputTimingFile = cms.string("muTimingModule.xml")
#process.simBayesMuCorrelatorTrackProducer.timingModuleFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/muTimingModuleTest.xml")#

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.L1TMuonSeq = cms.Sequence( #process.esProd +         
                                   process.simBayesMuCorrelatorTrackProducer 
                                   #+ process.dumpED
                                   #+ process.dumpES
)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

# process.out = cms.OutputModule("PoolOutputModule", 
#    fileName = cms.untracked.string("l1tomtf_superprimitives1.root")
# )
#process.output_step = cms.EndPath(process.out)

############################################################

process.omtfTTAnalyzer= cms.EDAnalyzer("MuCorrelatorAnalyzer", 
                                 outRootFile = cms.string("muCorrelatorTTAnalysis1.root"),
                                 etaCutFrom = cms.double(0.), #OMTF eta range
                                 etaCutTo = cms.double(2.4),
                                          
                                       MyProcess = cms.int32(1),
                                       DebugMode = cms.bool(verbose),      # printout lots of debug statements
                                       SaveAllTracks = cms.bool(True),   # save *all* L1 tracks, not just truth matched to primary particle
                                       SaveStubs = cms.bool(False),      # save some info for *all* stubs
                                       LooseMatch = cms.bool(True),     # turn on to use "loose" MC truth association
                                       L1Tk_nPar = cms.int32(4),         # use 4 or 5-parameter L1 track fit ??
                                       L1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
                                       TP_minNStub = cms.int32(4),       # require TP to have >= X number of stubs associated with it
                                       TP_minNStubLayer = cms.int32(4),  # require TP to have stubs in >= X layers/disks
                                       TP_minPt = cms.double(2.0),       # only save TPs with pt > X GeV
                                       TP_maxEta = cms.double(2.4),      # only save TPs with |eta| < X
                                       TP_maxZ0 = cms.double(30.0),      # only save TPs with |z0| < X cm
                                       L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),               ## TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"), ## MCTruth input 
                                       # other input collections
                                       L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                       MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                       MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                       TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       
                                       muCandQualityCut = cms.int32(12),
                                       analysisType = cms.string("withTrackPart")
                                        )
process.omtfTTAnalyzerPath = cms.Path(process.omtfTTAnalyzer)

###########################################################

############################################################

# use this if you want to re-run the stub making
#process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.TTTracksWithTruth,process.ana)

# use this if cluster/stub associators not available
# process.TTClusterStub, process.TTTracks, 
process.schedule = cms.Schedule(process.TTTracksWithTruth, process.L1TMuonPath, process.omtfTTAnalyzerPath) # 

# use this to only run tracking + track associator
#process.schedule = cms.Schedule(process.TTTracksWithTruth,process.ana)

#process.schedule.extend([process.output_step])
