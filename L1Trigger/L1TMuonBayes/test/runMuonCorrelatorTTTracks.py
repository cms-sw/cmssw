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
                                               'muCorrelatorEventPrint'
                    ),
       categories        = cms.untracked.vstring('l1tMuBayesEventPrint'),
       muCorrelatorEventPrint = cms.untracked.PSet(    
                         extension = cms.untracked.string('.txt'),                
                         threshold = cms.untracked.string('DEBUG'),
                         default = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
                         #INFO   =  cms.untracked.int32(0),
                         #DEBUG   = cms.untracked.int32(0),
                         l1tMuBayesEventPrint = cms.untracked.PSet( limit = cms.untracked.int32(100000000) )
                       ),
       debugModules = cms.untracked.vstring('L1TMuonBayesMuCorrelatorTrackProducer', 'OmtfTTAnalyzer', 'simOmtfDigis', 'omtfTTAnalyzer', 'simBayesMuCorrelatorTrackProducer') 
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

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '') 

############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

if GEOMETRY == "D17":
    Source_Files = cms.untracked.vstring(
          'file:///eos/user/k/kbunkow/cms_data/mc/PhaseIIFall17D/ZMM_EE29AF8E-51AF-E811-A2BD-484D7E8DF0D3_dump1000Events.root' 
)
elif GEOMETRY == "TkOnly":
    Source_Files = cms.untracked.vstring(
    "file:MuMinus_1to10_TkOnly.root"
)
else: 
    print "not a valid geometry!"

process.source = cms.Source("PoolSource", fileNames = Source_Files,
        inputCommands=cms.untracked.vstring(
        'keep *',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
        'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016s_simEmtfDigis__HLT')
)

#process.TFileService = cms.Service("TFileService", fileName = cms.string('muCorrelatorTTAnalysis1.root'), closeFileFast = cms.untracked.bool(True))


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

process.simBayesMuCorrelatorTrackProducer.ttTracksSource = cms.string("L1_TRACKER")
#process.simBayesMuCorrelatorTrackProducer.pdfModuleFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/pdfModule.xml")

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

# use this if you want to re-run the stub making
#process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.TTTracksWithTruth,process.ana)

# use this if cluster/stub associators not available 
#process.schedule = cms.Schedule(process.TTClusterStubTruth, process.TTTracksWithTruth, process.L1TMuonPath)
process.schedule = cms.Schedule(process.TTTracksWithTruth, process.L1TMuonPath)

# use this to only run tracking + track associator
#process.schedule = cms.Schedule(process.TTTracksWithTruth,process.ana)

#process.schedule.extend([process.output_step])
