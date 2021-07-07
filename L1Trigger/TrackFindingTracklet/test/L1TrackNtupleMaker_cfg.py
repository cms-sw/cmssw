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
# Set L1 tracking algorithm: 
# 'HYBRID' (baseline, 4par fit) or 'HYBRID_DISPLACED' (extended, 5par fit). 
# (Or legacy algos 'TMTT' or 'TRACKLET').
L1TRKALGO = 'HYBRID'  

WRITE_DATA = False

############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.L1track = dict(limit = -1)
process.MessageLogger.Tracklet = dict(limit = -1)

if GEOMETRY == "D49": 
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
elif GEOMETRY == "D76": 
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2026D76Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D76_cff')
else:
    print "this is not a valid geometry!!!"

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

#--- To use MCsamples scripts, defining functions get*data*(), 
#--- follow instructions https://cernbox.cern.ch/index.php/s/enCnnfUZ4cpK7mT

#from MCsamples.Scripts.getCMSdata_cfi import *
#from MCsamples.Scripts.getCMSlocaldata_cfi import *

if GEOMETRY == "D49":
  # Read data from card files (defines getCMSdataFromCards()):
  #from MCsamples.RelVal_1120.PU200_TTbar_14TeV_cfi import *
  #inputMC = getCMSdataFromCards()

  # Or read .root files from directory on local computer:
  #dirName = "$myDir/whatever/"
  #inputMC=getCMSlocaldata(dirName)

  # Or read specified dataset (accesses CMS DB, so use this method only occasionally):
  #dataName="/RelValTTbar_14TeV/CMSSW_11_2_0_pre5-PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/GEN-SIM-DIGI-RAW"
  #inputMC=getCMSdata(dataName)

  # Or read specified .root file:
  inputMC = ["/store/relval/CMSSW_11_3_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/00000/00260a30-734a-4a3a-a4b0-f836ce5502c6.root"] 

elif GEOMETRY == "D76":
  inputMC = ["/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/00026541-6200-4eed-b6f8-d3a1fd720e9c.root"]
else:
  print "this is not a valid geometry!!!"    
    
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC))

process.TFileService = cms.Service("TFileService", fileName = cms.string('TTbar_PU200_'+GEOMETRY+'.root'), closeFileFast = cms.untracked.bool(True))
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))


############################################################
# L1 tracking: stubs / DTC emulation
############################################################

process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')

# remake stubs? 
#from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
#process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")

#from SimTracker.TrackTriggerAssociation.TTClusterAssociation_cfi import *
#TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")

#process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
#process.TTClusterStubTruth = cms.Path(process.TrackTriggerAssociatorClustersStubs) 


# DTC emulation
process.load('L1Trigger.TrackerDTC.ProducerES_cff')
process.load('L1Trigger.TrackerDTC.ProducerED_cff')

# load code that analyzes DTCStubs
#process.load('L1Trigger.TrackerDTC.Analyzer_cff')

# modify default cuts
#process.TrackTriggerSetup.FrontEnd.BendCut = 5.0
#process.TrackTriggerSetup.Hybrid.MinPt = 1.0

process.dtc = cms.Path(process.TrackerDTCProducer)#*process.TrackerDTCAnalyzer)


############################################################
# L1 tracking
############################################################

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")

# HYBRID: prompt tracking
if (L1TRKALGO == 'HYBRID'):
    process.TTTracksEmulation = cms.Path(process.L1HybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1HybridTracksWithAssociators)
    NHELIXPAR = 4
    L1TRK_NAME  = "TTTracksFromTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_DISPLACED'):
    process.TTTracksEmulation = cms.Path(process.L1ExtendedHybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1ExtendedHybridTracksWithAssociators)
    NHELIXPAR = 5
    L1TRK_NAME  = "TTTracksFromExtendedTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigisExtended"
    
# LEGACY ALGORITHM (EXPERTS ONLY): TRACKLET  
elif (L1TRKALGO == 'TRACKLET'):
    print "\n WARNING: This is not the baseline algorithm! Prefer HYBRID or HYBRID_DISPLACED!"
    print "\n To run the Tracklet-only algorithm, ensure you have commented out 'CXXFLAGS=-DUSEHYBRID' in BuildFile.xml & recompiled! \n"
    process.TTTracksEmulation = cms.Path(process.L1HybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1HybridTracksWithAssociators)
    NHELIXPAR = 4
    L1TRK_NAME  = "TTTracksFromTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"

# LEGACY ALGORITHM (EXPERTS ONLY): TMTT  
elif (L1TRKALGO == 'TMTT'):
    print "\n WARNING: This is not the baseline algorithm! Prefer HYBRID or HYBRID_DISPLACED! \n"
    process.load("L1Trigger.TrackFindingTMTT.TMTrackProducer_Ultimate_cff")
    L1TRK_PROC  =  process.TMTrackProducer
    L1TRK_NAME  = "TMTrackProducer"
    L1TRK_LABEL = "TML1TracksKF4ParamsComb"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"
    NHELIXPAR = 4
    L1TRK_PROC.EnableMCtruth = cms.bool(False) # Reduce CPU use by disabling internal histos.
    L1TRK_PROC.EnableHistos  = cms.bool(False)
    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
    process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")
    process.TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag( cms.InputTag(L1TRK_NAME, L1TRK_LABEL) )
    process.TTTracksEmulation = cms.Path(process.offlineBeamSpot*L1TRK_PROC)
    process.TTTracksEmulationWithTruth = cms.Path(process.offlineBeamSpot*L1TRK_PROC*process.TrackTriggerAssociatorTracks)

else:
    print "ERROR: Unknown L1TRKALGO option"
    exit(1)


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
                                       L1TrackInputTag = cms.InputTag(L1TRK_NAME, L1TRK_LABEL),         # TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag(L1TRUTH_NAME, L1TRK_LABEL),  # MCTruth input 
                                       # other input collections
                                       L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                       MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                       MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                       TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                       # tracking in jets (--> requires AK4 genjet collection present!)
                                       TrackingInJets = cms.bool(False),
                                       GenJetInputTag = cms.InputTag("ak4GenJets", "")
                                       )

process.ana = cms.Path(process.L1TrackNtuple)


############################################################
# final schedule of what is to be run
############################################################

# use this if you want to re-run the stub making
# process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.dtc,process.TTTracksEmulationWithTruth,process.ana)

# use this if cluster/stub associators not available 
# process.schedule = cms.Schedule(process.TTClusterStubTruth,process.dtc,process.TTTracksEmulationWithTruth,process.ana)

# use this to only run tracking + track associator
process.schedule = cms.Schedule(process.dtc,process.TTTracksEmulationWithTruth,process.ana)


############################################################
# write output dataset?
############################################################

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
  process.writeDataset.outputCommands.append('keep  *TTTrack*_*_*_*')
  process.writeDataset.outputCommands.append('keep  *TTStub*_*_*_*')

  process.pd = cms.EndPath(process.writeDataset)
  process.schedule.append(process.pd)
  
