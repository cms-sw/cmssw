############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import os
process = cms.Process("L1TrackNtuple")
 
 
############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
#process.load('L1Trigger.TrackTrigger.TkOnlyTiltedGeom_cff') # Tracker-only geometry, TILTED
#process.load('L1Trigger.TrackTrigger.TkOnlyFlatGeom_cff')   # Tracker-only geometry, FLAT
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff') # D4 CMS geometry, corresponds to tilted tracker
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')     # D4 CMS geometry, corresponds to tilted tracker

process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
#process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))
Source_Files = cms.untracked.vstring(
## PU=0
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/029B633F-16FD-E611-B79B-0025905B8564.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/44773108-16FD-E611-868E-0CC47A4C8ECA.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/7271F39A-16FD-E611-87FB-0CC47A7C3636.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/7A682CC9-14FD-E611-8835-0CC47A78A33E.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/88624741-15FD-E611-A16F-0CC47A7C3636.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/8C34EF40-15FD-E611-9827-0CC47A78A4A6.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/A635A721-16FD-E611-ACFC-0025905A6060.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/B2BE63BC-15FD-E611-BEEE-0025905B85BE.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/BE39F598-16FD-E611-B30B-0CC47A78A4A6.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/C0A60D3F-16FD-E611-8E6A-0025905A6066.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4T-v1/00000/C49218BD-15FD-E611-9F0A-0025905B858A.root"
#
## PU=140 
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/CEF28581-5BFD-E611-99D2-0CC47A7C34E6.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/D0A888FE-5AFD-E611-993C-0CC47A745282.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/D8A72F7E-5AFD-E611-8229-0025905A6088.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/E02AD679-5AFD-E611-B244-0CC47A4C8F1C.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/E240C77F-5AFD-E611-A6E9-0CC47A78A418.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/E42E653E-5EFD-E611-A485-0025905B8560.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/E4C327C3-59FD-E611-99CA-0025905B85FC.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/E8A70F7A-5AFD-E611-829C-0CC47A4D76C8.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/EA12D2AF-59FD-E611-B53A-0CC47A4C8E1E.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/ECF0A7B3-59FD-E611-B50D-0025905B8568.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/EEC6D984-5BFD-E611-AC38-0025905A60B2.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/F0404FB3-59FD-E611-B261-0CC47A4C8E26.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/F2123EB4-59FD-E611-ABE4-0025905B85F6.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/F2F0BB7A-5AFD-E611-94CD-0CC47A4D767A.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/F2F55AAB-67FD-E611-8E5A-0CC47A7C340C.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/F82C0AFE-5AFD-E611-8BA5-0CC47A4D75F0.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/FA53527E-5AFD-E611-A9AC-0CC47A78A426.root",
#"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2023_realistic_v4_D4TPU140-v1/00000/FA761BA8-58FD-E611-8FFC-0CC47A4D7678.root"
    )
process.source = cms.Source("PoolSource", fileNames = Source_Files)

process.TFileService = cms.Service("TFileService", fileName = cms.string('MuPt10_PU0.root'), closeFileFast = cms.untracked.bool(True))


############################################################
# Path definitions & schedule
############################################################

# beamspot 
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.BS = cms.Path(process.offlineBeamSpot)
 
# run cluster+stubs associators
#process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
process.TTClusterStubAssociator = cms.Path(process.TrackTriggerAssociatorClustersStubs)

process.TTTracks = cms.EDProducer("L1TrackProducer",
				 SimTrackSource = cms.InputTag("g4SimHits"),
				 SimVertexSource = cms.InputTag("g4SimHits"),
				 TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
				 TTStubMCTruthSource = cms.InputTag("TTStubAssociatorFromPixelDigis","StubAccepted"),
                 BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                 asciiFileName = cms.untracked.string("evlist_debugstub.txt"),
    )
process.TrackTriggerTTTracks = cms.Sequence(process.TTTracks)
process.TT_step = cms.Path(process.TrackTriggerTTTracks)
process.TTAssociator = cms.Path(process.TrackTriggerAssociatorTracks)

### integer emulation 
process.TTTracksInteger = cms.EDProducer("L1FPGATrackProducer",
                                         # general L1 tracking inputs
                                         SimTrackSource = cms.InputTag("g4SimHits"),
                                         SimVertexSource = cms.InputTag("g4SimHits"),
                                         TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                         TTStubMCTruthSource = cms.InputTag("TTStubAssociatorFromPixelDigis","StubAccepted"),
                                         BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                         # specific emulation inputs
                                         fitPatternFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/test/fitpattern.txt'),
                                         memoryModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/test/memorymodules_new.dat'),
                                         processingModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/test/processingmodules_new.dat'),
                                         wiresFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/test/wires_new.dat')
    )
process.TrackTriggerTTTracksInteger = cms.Sequence(process.TTTracksInteger)
process.TT_step_Integer = cms.Path(process.TrackTriggerTTTracksInteger)


process.TrackTriggerAssociatorInteger = process.TTTrackAssociatorFromPixelDigis.clone()
process.TrackTriggerAssociatorInteger.TTTracks = cms.VInputTag( cms.InputTag("TTTracksInteger", "Level1TTTracks") )
process.TrackTriggerAssociatorTracksInteger = cms.Sequence( process.TrackTriggerAssociatorInteger )
process.TTAssociator_Integer = cms.Path(process.TrackTriggerAssociatorTracksInteger)


############################################################
# Define the track ntuple process, MyProcess is the (unsigned) PDGID corresponding to the process which is run
# e.g. single electron/positron = 11
#      single pion+/pion- = 211
#      single muon+/muon- = 13 
#      pions in jets = 6
#      taus = 15
#      all TPs = 1
############################################################

process.L1TrackNtuple = cms.EDAnalyzer('L1TrackNtupleMaker',
                                       MyProcess = cms.int32(13),
                                       DebugMode = cms.bool(False),      # printout lots of debug statements
                                       SaveAllTracks = cms.bool(True),   # save *all* L1 tracks, not just truth matched to primary particle
                                       SaveStubs = cms.bool(False),      # save some info for *all* stubs
                                       L1Tk_nPar = cms.int32(4),         # use 4 or 5-parameter L1 track fit ??
                                       L1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
                                       TP_minNStub = cms.int32(4),       # require TP to have >= X number of stubs associated with it
                                       TP_minNStubLayer = cms.int32(4),  # require TP to have stubs in >= X layers/disks
                                       TP_minPt = cms.double(2.0),       # only save TPs with pt > X GeV
                                       TP_maxEta = cms.double(2.4),      # only save TPs with |eta| < X
                                       TP_maxZ0 = cms.double(30.0),      # only save TPs with |z0| < X cm
                                       # floating point input
                                       L1TrackInputTag = cms.InputTag("TTTracks", "Level1TTTracks"),               ## TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"), ## MCTruth input 
                                       # emulation input
                                       #L1TrackInputTag = cms.InputTag("TTTracksInteger", "Level1TTTracks"),               ## TTTrack input
                                       #MCTruthTrackInputTag = cms.InputTag("TrackTriggerAssociatorInteger", "Level1TTTracks"), ## MCTruth input 
                                       # other input collections
                                       MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                       MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                       TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       )
process.ana = cms.Path(process.L1TrackNtuple)

#output module (can use this to store edm-file instead of root-ntuple)
process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string("Tracks_SingleMuon_TILTED.root"),
                                fastCloning = cms.untracked.bool( False ),
                                outputCommands = cms.untracked.vstring('keep *')
)
process.FEVToutput_step = cms.EndPath(process.out)


# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023tilted
#process = cust_2023tilted(process)
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023LReco
#process = cust_2023LReco(process)


## floating-point version 
process.schedule = cms.Schedule(process.TTClusterStubAssociator,process.BS,process.TT_step,process.TTAssociator,process.ana)

## integer emulation version
#process.schedule = cms.Schedule(process.TTClusterStubAssociator,process.BS,process.TT_step_Integer,process.TTAssociator_Integer,process.ana)

