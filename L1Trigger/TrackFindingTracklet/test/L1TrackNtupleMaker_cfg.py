############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import os
process = cms.Process("L1TrackNtuple")

#GEOMETRY = "D21"
GEOMETRY = "D41"

 
############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

if GEOMETRY == "D17": 
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
elif GEOMETRY == "D21": 
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2023D21_cff')
elif GEOMETRY == "D41":
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2023D41Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2023D41_cff')
elif GEOMETRY == "TkOnly": 
    print "using standalone tilted (T5) tracker geometry" 
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

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

if GEOMETRY == "D17": # Tilted barrel T5 tracker
    Source_Files = cms.untracked.vstring(
        "/store/relval/CMSSW_9_3_7/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/5A8CFF7F-1E2D-E811-A7B0-0242AC130002.root"
    )
elif GEOMETRY == "D21": # Tilted barrel T6 tracker
    Source_Files = cms.untracked.vstring(
        "/store/relval/CMSSW_10_4_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_103X_upgrade2023_realistic_v2_2023D21PU200-v1/20000/FFF48AB4-E5E6-3842-8A5B-20E2B7E497BC.root"
    )
elif GEOMETRY == "D41": # Tilted barrel T14 tracker
    Source_Files = cms.untracked.vstring(
        "/store/relval/CMSSW_10_6_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_106X_upgrade2023_realistic_v2_2023D41PU200-v1/10000/FEA5D564-937A-8D4B-9C9A-696EFC05AB58.root"
    )
elif GEOMETRY == "TkOnly":
    Source_Files = cms.untracked.vstring(
        "file:/afs/cern.ch/work/s/skinnari/public/L1TK_90X/MuMinus_1to10_TkOnly.root"
)
else: 
    print "not a valid geometry!"

process.source = cms.Source("PoolSource", 
                            fileNames = Source_Files,
                            inputCommands = cms.untracked.vstring(
                              'keep *_*_*_*',
                              'drop l1tEMTFHit2016*_*_*_*',
                              'drop l1tEMTFTrack2016*_*_*_*'
                              )
                            )

process.TFileService = cms.Service("TFileService", fileName = cms.string('TTbar_PU200_'+GEOMETRY+'.root'), closeFileFast = cms.untracked.bool(True))



############################################################
# L1 tracking
############################################################

# remake stubs ?
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")

if GEOMETRY != "TkOnly":
    from SimTracker.TrackTriggerAssociation.TTClusterAssociation_cfi import *
    TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")
    
process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
process.TTClusterStubTruth = cms.Path(process.TrackTriggerAssociatorClustersStubs)

from L1Trigger.TrackFindingTracklet.Tracklet_cfi import *

### floating-point simulation
#process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
#TTTracksFromTracklet.asciiFileName = cms.untracked.string("output.txt")
#process.TTTracks = cms.Path(process.L1TrackletTracks)
#process.TTTracksWithTruth = cms.Path(process.L1TrackletTracksWithAssociators)

## emulation 
process.load("L1Trigger.TrackFindingTracklet.L1TrackletEmulationTracks_cff")
process.TTTracksEmulation = cms.Path(process.L1TrackletEmulationTracks)
process.TTTracksEmulationWithTruth = cms.Path(process.L1TrackletEmulationTracksWithAssociators)


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
                                       MyProcess = cms.int32(1),
                                       DebugMode = cms.bool(False),      # printout lots of debug statements
                                       SaveAllTracks = cms.bool(True),   # save *all* L1 tracks, not just truth matched to primary particle
                                       SaveStubs = cms.bool(False),      # save some info for *all* stubs
                                       L1Tk_nPar = cms.int32(4),         # use 4 or 5-parameter L1 track fit ??
                                       L1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
                                       TP_minNStub = cms.int32(4),       # require TP to have >= X number of stubs associated with it
                                       TP_minNStubLayer = cms.int32(4),  # require TP to have stubs in >= X layers/disks
                                       TP_minPt = cms.double(2.0),       # only save TPs with pt > X GeV
                                       TP_maxEta = cms.double(2.5),      # only save TPs with |eta| < X
                                       TP_maxZ0 = cms.double(30.0),      # only save TPs with |z0| < X cm
                                       #L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),                 ## TTTrack input
                                       L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),         ## TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"),  ## MCTruth input 
                                       # other input collections
                                       L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                       MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                       MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                       TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       ## tracking in jets stuff (--> requires AK4 genjet collection present!)
                                       TrackingInJets = cms.bool(True),
                                       GenJetInputTag = cms.InputTag("ak4GenJets", ""),
                                       )
process.ana = cms.Path(process.L1TrackNtuple)

# use this if you want to re-run the stub making
#process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.TTTracksEmulationWithTruth,process.ana)

# use this if cluster/stub associators not available 
#process.schedule = cms.Schedule(process.TTClusterStubTruth,process.TTTracksEmulationWithTruth,process.ana)

# use this to only run tracking + track associator
#process.schedule = cms.Schedule(process.TTTracksWithTruth,process.ana)            # floating-point simulation
process.schedule = cms.Schedule(process.TTTracksEmulationWithTruth,process.ana)    # emulation
