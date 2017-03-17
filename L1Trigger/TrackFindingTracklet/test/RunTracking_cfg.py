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
#process.load('L1Trigger.TrackTrigger.TkOnlyTiltedGeom_cff') # TILTED
#process.load('L1Trigger.TrackTrigger.TkOnlyFlatGeom_cff') # FLAT
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')

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
#    "/store/relval/CMSSW_9_0_0_pre5/RelValBsToPhiPhi4K_14TeV/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4TnoVAL-v1/00000/40DDE577-EE02-E711-AA5D-0CC47A78A4BA.root",
#    "/store/relval/CMSSW_9_0_0_pre5/RelValBsToPhiPhi4K_14TeV/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D4TnoVAL-v1/00000/BE0EEA78-EE02-E711-81F9-0CC47A78A446.root"
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D11-v1/00000/0C39AAB5-28FE-E611-839D-0025905B8600.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D11-v1/00000/36D355B4-28FE-E611-971F-0CC47A4D7626.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D11-v1/00000/3CAA1BEE-28FE-E611-A068-0CC47A4C8F06.root",
"/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D11-v1/00000/6EAF64B8-29FE-E611-9AD8-0025905A48D8.root"
    )
process.source = cms.Source("PoolSource", fileNames = Source_Files)

process.TFileService = cms.Service("TFileService", fileName = cms.string('Ntuple_TEST_MuPt10.root'), closeFileFast = cms.untracked.bool(True))


############################################################
# Path definitions & schedule
############################################################

# run cluster+stubs associators
process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
process.TTClusterStubAssociator = cms.Path(process.TrackTriggerAssociatorClustersStubs)


#run the tracking
#BeamSpotFromSim = cms.EDProducer("BeamSpotFromSimProducer")
#process.TT_step = cms.Path(process.TrackTriggerTTTracks)
#process.TTAssociator_step = cms.Path(process.TrackTriggerAssociatorTracks)


process.TTTracks = cms.EDProducer("L1TrackProducer",
				 SimTrackSource = cms.InputTag("g4SimHits"),
				 SimVertexSource = cms.InputTag("g4SimHits"),
				 TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
				 TTStubMCTruthSource = cms.InputTag("TTStubAssociatorFromPixelDigis","StubAccepted"),
#                 asciiFileName = cms.untracked.string("evlist_oneelectron.txt"),
    )
#process.TrackTriggerTTTracks = cms.Sequence(process.BeamSpotFromSim*process.TTTracks)
process.TrackTriggerTTTracks = cms.Sequence(process.TTTracks)

process.TT_step = cms.Path(process.TrackTriggerTTTracks)



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
                                       L1TrackInputTag = cms.InputTag("TTTracks", "Level1TTTracks"),               ## TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag("", ""), ## MCTruth input 
                                       MCTruthClusterInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "ClusterAccepted"),
                                       MCTruthStubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
                                       TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       )
process.ana = cms.Path(process.L1TrackNtuple)

#output module (can use this to store edm-file instead of root-ntuple)
#process.out = cms.OutputModule( "PoolOutputModule",
#                                fileName = cms.untracked.string("Tracks_SingleMuon_TILTED.root"),
#                                fastCloning = cms.untracked.bool( False ),
#                                outputCommands = cms.untracked.vstring('keep *')
#)
#process.FEVToutput_step = cms.EndPath(process.out)


# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023TTI
#process = cust_2023TTI(process)

#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023tilted
#process = cust_2023tilted(process)
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023LReco
#process = cust_2023LReco(process)


#process.schedule = cms.Schedule(process.TT_step,process.TTAssociator_step,process.ana)
process.schedule = cms.Schedule(process.TT_step,process.ana)
#process.schedule = cms.Schedule(process.TT_step,process.FEVToutput_step)
#process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubAssociator,process.TT_step,process.ana)
#process.schedule = cms.Schedule(process.TT_step)

