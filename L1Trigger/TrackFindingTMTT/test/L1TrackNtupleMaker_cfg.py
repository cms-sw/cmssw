############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import os
process = cms.Process("L1TrackNtuple")


############################################################
# edit options here
############################################################

GEOMETRY = "D49"

# Specify L1 tracking algo ('HYBRID', 'HYBRID_DISPLACED', 'TMTT','HYBRID_FLOAT', 'TRACKLET_FLOAT'),
# (To run Tracklet, set L1TRKALGO='HYBRID', and set USEHYBRID in TrackFindingTracklet/interface/Settings.h)
L1TRKALGO = 'TMTT'

# Write output dataset?
WRITE_DATA = False

if (L1TRKALGO == 'HYBRID_FLOAT'):
    if ( not os.path.exists( os.environ['CMSSW_BASE']+'/src/L1Trigger/HybridFloat' ) ):
        print "=== ERROR: Please checkout HybridFloat code from git before using this option ==="; exit


############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.L1track=dict()
process.MessageLogger.Tracklet=dict()
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

if GEOMETRY == "D49": 
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
else:
    print "this is not a valid geometry!!!"
    exit

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(50))

# Get list of MC datasets from repo, or specify yourself.

def getTxtFile(txtFileName): 
  return FileUtils.loadListFromFile(os.environ['CMSSW_BASE']+'/src/L1Trigger/TrackFindingTMTT/data/MCsamples/'+txtFileName)

# MC txt files are in https://github.com/cms-data/L1Trigger-TrackFindingTMTT.
if GEOMETRY == "D49":
  inputMC = getTxtFile('1110/RelVal/TTbar/PU200.txt')

else:
    print "this is not a valid geometry!!!"
    
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(*inputMC),
                            inputCommands = cms.untracked.vstring(
                              'keep *_*_*_*',
                              'drop l1tEMTFHit2016*_*_*_*',
                              'drop l1tEMTFTrack2016*_*_*_*'
                              )
                            )

process.TFileService = cms.Service("TFileService", fileName = cms.string('TTbar_PU200_'+L1TRKALGO+'.root'), closeFileFast = cms.untracked.bool(True))

process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))


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

NHELIXPAR = 4
if   (L1TRKALGO == 'HYBRID'):
    process.load("L1Trigger.TrackFindingTracklet.Tracklet_cfi") 
    L1TRK_PROC  =  process.TTTracksFromTrackletEmulation
    L1TRK_NAME  = "TTTracksFromTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
elif (L1TRKALGO == 'HYBRID_DISPLACED'):
    process.load("L1Trigger.TrackFindingTracklet.Tracklet_cfi") 
    L1TRK_PROC  =  process.TTTracksFromExtendedTrackletEmulation
    L1TRK_NAME  = "TTTracksFromExtendedTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    NHELIXPAR = 5
elif (L1TRKALGO == 'TMTT'):
    process.load("L1Trigger.TrackFindingTMTT.TMTrackProducer_Ultimate_cff")
    L1TRK_PROC  =  process.TMTrackProducer
    L1TRK_NAME  = "TMTrackProducer"
    L1TRK_LABEL = "TML1TracksKF4ParamsComb"
    L1TRK_PROC.EnableMCtruth = cms.bool(False) # Reduce CPU use by disabling internal histos.
    L1TRK_PROC.EnableHistos  = cms.bool(False)
elif (L1TRKALGO == 'HYBRID_FLOAT'):
    process.load("L1Trigger.HybridFloat.HybridTrackProducer_cff")
    L1TRK_PROC  =  process.HybridTrackProducer
    L1TRK_NAME  = "HybridTrackProducer"
    L1TRK_LABEL = "HybridL1TracksKF4ParamsComb"
    L1TRK_PROC.EnableMCtruth = cms.bool(False) # Reduce CPU use by disabling internal histos.
    L1TRK_PROC.EnableHistos  = cms.bool(False)
elif (L1TRKALGO == 'TRACKLET_FLOAT'):
    process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
    L1TRK_PROC  =  process.TTTracksFromTracklet
    L1TRK_NAME  = "TTTracksFromTracklet"
    L1TRK_LABEL = "Level1TTTracks"
else:
    print "ERROR: Unknown L1TRKALGO option"
    exit(1)

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")
process.TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag( cms.InputTag(L1TRK_NAME, L1TRK_LABEL) )

## emulation 
process.TTTracksEmulation = cms.Path(process.offlineBeamSpot*L1TRK_PROC)
process.TTTracksEmulationWithTruth = cms.Path(process.offlineBeamSpot*L1TRK_PROC*process.TrackTriggerAssociatorTracks)


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
                                       L1Tk_nPar = cms.int32(NHELIXPAR), # use 4 or 5-parameter L1 tracking?
                                       L1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
                                       TP_minNStub = cms.int32(4),       # require TP to have >= X number of stubs associated with it
                                       TP_minNStubLayer = cms.int32(4),  # require TP to have stubs in >= X layers/disks
                                       TP_minPt = cms.double(2.0),       # only save TPs with pt > X GeV
                                       TP_maxEta = cms.double(2.5),      # only save TPs with |eta| < X
                                       TP_maxZ0 = cms.double(30.0),      # only save TPs with |z0| < X cm
                                       L1TrackInputTag = cms.InputTag(L1TRK_NAME, L1TRK_LABEL), # TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", L1TRK_LABEL),  ## MCTruth input 
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
# process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.TTTracksEmulationWithTruth,process.ana)

# use this if cluster/stub associators not available 
# process.schedule = cms.Schedule(process.TTClusterStubTruth,process.TTTracksEmulationWithTruth,process.ana)

# use this to only run tracking + track associator
process.schedule = cms.Schedule(process.TTTracksEmulationWithTruth,process.ana)

if (WRITE_DATA):

  process.writeDataset = cms.OutputModule("PoolOutputModule",
      splitLevel = cms.untracked.int32(0),
      eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
      outputCommands = process.RAWSIMEventContent.outputCommands,
      fileName = cms.untracked.string('output_dataset.root'), ## ADAPT IT ##
      dataset = cms.untracked.PSet(
          filterName = cms.untracked.string(''),
          dataTier = cms.untracked.string('GEN-SIM')
      )
  )
  # Include TMTT L1 tracks & associators + stubs.
  process.writeDataset.outputCommands.append('keep  *TTTrack*_*_*_*')
  process.writeDataset.outputCommands.append('keep  *TTStub*_*_*_*')

  process.pd = cms.EndPath(process.writeDataset)
  process.schedule.append(process.pd)

