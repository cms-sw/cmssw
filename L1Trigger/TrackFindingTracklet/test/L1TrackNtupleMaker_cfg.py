# N,B, DUE TO THE CHANGE IN STUB WINDOW SIZES WITH CMSSW 14_2_0_PRE2, THIS JOB HAS BEEN NODIFIED TO
# RECREATE THE STUBS, WHICH IS NECESSARY WHEN RUNNING ON MONTE CARLO GENERATED WITH OLDER VERSIONS.

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

# D88 was used for CMSSW_12_6 datasets, and D98 recommended for more recent ones.
#GEOMETRY = "D88"
GEOMETRY = "D98"

# Set L1 tracking algorithm:
# 'HYBRID' (baseline, 4par fit) or 'HYBRID_DISPLACED' (extended, 5par fit).
# 'HYBRID_NEWKF' (baseline, 4par fit, with bit-accurate KF emulation),
# 'HYBRID_REDUCED' to use the "Summer Chain" configuration with reduced inputs.
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
process.MessageLogger.TrackTriggerHPH = dict(limit = -1)

if GEOMETRY == "D88" or GEOMETRY == 'D98':
    print("using geometry " + GEOMETRY + " (tilted)")
    process.load('Configuration.Geometry.GeometryExtendedRun4' + GEOMETRY + 'Reco_cff')
    process.load('Configuration.Geometry.GeometryExtendedRun4' + GEOMETRY +'_cff')
else:
    print("this is not a valid geometry!!!")

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
# Change needed to run with D98 geometry in recent CMSSW versions.
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

#--- To use MCsamples scripts, defining functions get*data*() for easy MC access,
#--- follow instructions in https://github.com/cms-L1TK/MCsamples

#from MCsamples.Scripts.getCMSdata_cfi import *
#from MCsamples.Scripts.getCMSlocaldata_cfi import *

if GEOMETRY == "D98":

  # Read data from card files (defines getCMSdataFromCards()):
  #from MCsamples.RelVal_1400_D98.PU200_TTbar_14TeV_cfi import *
  #inputMC = getCMSdataFromCards()

  # Or read .root files from directory on local computer:
  #dirName = "$scratchmc/MCsamples1400_D98/RelVal/TTbar/PU0/"
  #inputMC=getCMSlocaldata(dirName)

  # Or read specified dataset (accesses CMS DB, so use this method only occasionally):
  #dataName="/RelValTTbar_14TeV/CMSSW_14_0_0_pre2-PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/GEN-SIM-DIGI-RAW"
  #inputMC=getCMSdata(dataName)

  inputMC = ["/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0b2b0b0b-f312-48a8-9d46-ccbadc69bbfd.root"]

elif GEOMETRY == "D88":

  # Read specified .root file:
  inputMC = ["/store/mc/CMSSW_12_6_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_125X_mcRun4_realistic_v5_2026D88PU200RV183v2-v1/30000/0959f326-3f52-48d8-9fcf-65fc41de4e27.root"]

else:

  print("this is not a valid geometry!!!")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC))

#if GEOMETRY == "D76":
#  # If reading old MC dataset, drop incompatible EDProducts.
#  process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool(False)
#  process.source.inputCommands = cms.untracked.vstring()
#  process.source.inputCommands.append('keep  *_*_*Level1TTTracks*_*')
#  process.source.inputCommands.append('keep  *_*_*StubAccepted*_*')
#  process.source.inputCommands.append('keep  *_*_*ClusterAccepted*_*')
#  process.source.inputCommands.append('keep  *_*_*MergedTrackTruth*_*')
#  process.source.inputCommands.append('keep  *_genParticles_*_*')

# Use skipEvents to select particular single events for test vectors
#process.source.skipEvents = cms.untracked.uint32(11)

process.TFileService = cms.Service("TFileService", fileName = cms.string('L1TrkNtuple.root'), closeFileFast = cms.untracked.bool(True))
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


# load code that associates stubs with mctruth
process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
# DTC emulation
process.load('L1Trigger.TrackerDTC.DTC_cff')

# load code that analyzes DTCStubs
process.load('L1Trigger.TrackerDTC.Analyzer_cff')

# modify default cuts
#process.TrackTriggerSetup.FrontEnd.BendCut = 5.0
#process.TrackTriggerSetup.Hybrid.MinPt = 1.0

process.dtc = cms.Path(process.StubAssociator + process.ProducerDTC + process.AnalyzerDTC)

############################################################
# L1 tracking
############################################################

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")

# HYBRID: prompt tracking
if (L1TRKALGO == 'HYBRID'):
    process.TTTracksEmulation = cms.Path(process.L1THybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1THybridTracksWithAssociators)
    NHELIXPAR = 4
    L1TRK_NAME  = "l1tTTTracksFromTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_DISPLACED'):
    process.TTTracksEmulation = cms.Path(process.L1TExtendedHybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1TExtendedHybridTracksWithAssociators)
    NHELIXPAR = 5
    L1TRK_NAME  = "l1tTTTracksFromExtendedTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigisExtended"

# HYBRID_NEWKF: prompt tracking or reduced
elif (L1TRKALGO == 'HYBRID_NEWKF' or L1TRKALGO == 'HYBRID_REDUCED'):
    process.load( 'L1Trigger.TrackFindingTracklet.Producer_cff' )
    process.load( 'L1Trigger.TrackFindingTracklet.Analyzer_cff' )
    NHELIXPAR = 4
    L1TRK_NAME  = process.TrackFindingTrackletAnalyzer_params.OutputLabelTFP.value()
    L1TRK_LABEL = process.TrackFindingTrackletProducer_params.BranchTTTracks.value()
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"
    process.TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag( cms.InputTag(L1TRK_NAME, L1TRK_LABEL) )
    process.HybridNewKF = cms.Sequence(process.L1THybridTracks + process.ProducerTM + process.ProducerDR + process.ProducerKF + process.ProducerTQ + process.ProducerTFP)
    process.TTTracksEmulation = cms.Path(process.HybridNewKF)
    #process.TTTracksEmulationWithTruth = cms.Path(process.HybridNewKF +  process.TrackTriggerAssociatorTracks)
    # Optionally include code producing performance plots & end-of-job summary.
    process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
    process.TTTracksEmulationWithTruth = cms.Path(process.HybridNewKF +  process.TrackTriggerAssociatorTracks + process.StubAssociator +  process.AnalyzerTracklet + process.AnalyzerTM + process.AnalyzerDR + process.AnalyzerTQ + process.AnalyzerKF + process.AnalyzerTFP )
    from L1Trigger.TrackFindingTracklet.Customize_cff import *
    if (L1TRKALGO == 'HYBRID_NEWKF'):
        fwConfig( process )
    if (L1TRKALGO == 'HYBRID_REDUCED'):
        reducedConfig( process )
    # Needed by L1TrackNtupleMaker
    process.HitPatternHelperSetup.useNewKF = True

# LEGACY ALGORITHM (EXPERTS ONLY): TRACKLET
elif (L1TRKALGO == 'TRACKLET'):
    print("\n WARNING: This is not the baseline algorithm! Prefer HYBRID or HYBRID_DISPLACED!")
    print("\n To run the Tracklet-only algorithm, ensure you have commented out 'CXXFLAGS=-DUSEHYBRID' in BuildFile.xml & recompiled! \n")
    process.TTTracksEmulation = cms.Path(process.L1THybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1THybridTracksWithAssociators)
    from L1Trigger.TrackFindingTracklet.Customize_cff import *
    trackletConfig( process )
    NHELIXPAR = 4
    L1TRK_NAME  = "l1tTTTracksFromTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"

# LEGACY ALGORITHM (EXPERTS ONLY): TMTT
elif (L1TRKALGO == 'TMTT'):
    print("\n WARNING: This is not the baseline algorithm! Prefer HYBRID or HYBRID_DISPLACED! \n")
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
    print("ERROR: Unknown L1TRKALGO option")
    exit(1)


# Define L1 track ntuple maker
from L1Trigger.TrackFindingTracklet.L1TrackNtupleMaker_cfi import *
process.L1TrackNtuple = L1TrackNtupleMaker.clone(
   L1Tk_nPar = NHELIXPAR, # use 4 or 5-parameter L1 tracking?
   L1TrackInputTag = (L1TRK_NAME, L1TRK_LABEL),         # TTTrack input
   MCTruthTrackInputTag = (L1TRUTH_NAME, L1TRK_LABEL),  # MCTruth input
)

process.ana = cms.Path(process.L1TrackNtuple)


############################################################
# final schedule of what is to be run
############################################################

# use this if you want to re-run the stub making
#process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.dtc,process.TTTracksEmulationWithTruth,process.ana)

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



