# N.B., DUE TO THE CHANGE IN STUB WINDOW SIZES WITH CMSSW 14_2_0_PRE2, THIS JOB HAS BEEN NODIFIED TO
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

# D110 recommended (but D98 still works)
#GEOMETRY = "D98"
GEOMETRY = "D110"

# Set L1 tracking algorithm:
# 'HYBRID' (baseline, 4par fit) or 'HYBRID_DISPLACED' (extended, 5par fit).
# 'HYBRID_NEWKF' (baseline, 4par fit, with bit-accurate KF emulation),
# 'HYBRID_REDUCED' to use the "L5L6" seeding only reduced configuration.
# 'HYBRID_DISPLACED_SIM' displaced tracklet followed by DR simulation and 5 param fit sim
# (Or legacy algos 'TMTT' or 'TRACKLET').
L1TRKALGO = 'HYBRID_NEWKF'
TRAINING_RUN = True

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
process.MessageLogger.cout.enableStatistics = True
process.MessageLogger.cerr.enableStatistics = True

print("using geometry " + GEOMETRY + " (tilted)")
process.load('Configuration.Geometry.GeometryExtendedRun4' + GEOMETRY + 'Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4' + GEOMETRY +'_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# Change needed to run with D98 geometry in recent CMSSW versions.
if GEOMETRY == 'D98':
    process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')
elif GEOMETRY == 'D110':
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
else:
    print("this is not a valid geometry!!!")

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

#--- To use MCsamples scripts, defining functions get*data*() for easy MC access,
#--- follow instructions in https://github.com/cms-L1TK/MCsamples

#from MCsamples.Scripts.getCMSdata_cfi import *
#from MCsamples.Scripts.getCMSlocaldata_cfi import *

if GEOMETRY == "D110":

  # Read data from card files (defines getCMSdataFromCards()):
  #from MCsamples.RelVal_1510_D110.PU200_TTbar_14TeV_cfi import *
  #inputMC = getCMSdataFromCards()

  # Or read .root files from directory on local computer:
  #dirName = "$scratchmc/MCsamples1510_D110/RelVal/TTbar/PU0/"
  #inputMC=getCMSlocaldata(dirName)  

  # Or read specified dataset (accesses CMS DB, so use this method only occasionally):
  #dataName="/RelValTTbar_14TeV_TuneCP5/CMSSW_15_1_0_pre5-PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v2/GEN-SIM-DIGI-RAW"
  #inputMC=getCMSdata(dataName)
  
  # ttbar + 200PU
  inputMC = [
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/a3240179-8245-45eb-8364-3ea6766bd349.root",
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/87e6ac25-289d-466e-9309-4ee69340b43e.root",
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/744fa2bd-3d41-4415-9ff9-75eff89e553d.root",
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/25127e07-9934-4e1b-9b79-d4e71ae467e1.root",
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/5f0060e6-522c-47bc-ba4c-2b95118d3d93.root",
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/ac5acc0b-7013-49b8-8c49-f6fc122184dc.root",
        "/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v1/2590000/4f6fd6e0-c1c2-47f2-b227-9f67dbbc5ecf.root"
    ]

elif GEOMETRY == "D98":

  # Or read .root files from directory on local computer:
  dirName = "$scratchmc/MCsamples1400_D98/RelVal/TTbar/PU200/"
  inputMC=getCMSlocaldata(dirName)  
  
  #  inputMC = ['/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0b2b0b0b-f312-48a8-9d46-ccbadc69bbfd.root']  
  
else:

  print("this is not a valid geometry!!!")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC))

# Drop previously reconstructed L1 tracks + their truth association to avoid risk of analysing them instead of new tracks created by this job.
process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool(False)
process.source.inputCommands = cms.untracked.vstring()
process.source.inputCommands.append('keep *_*_*_*')
process.source.inputCommands.append('drop  *_*_*Level1TTTracks*_*')

#if GEOMETRY == "D76":
#  # If reading old MC dataset, drop incompatible EDProducts.
#  process.source.inputCommands.append('drop *_*_*_*')
#  process.source.inputCommands.append('keep *_*_*Level1TTTracks*_*')
#  process.source.inputCommands.append('keep *_*_*StubAccepted*_*')
#  process.source.inputCommands.append('keep *_*_*ClusterAccepted*_*')
#  process.source.inputCommands.append('keep *_*_*MergedTrackTruth*_*')
#  process.source.inputCommands.append('keep *_genParticles_*_*')

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
# load code that analyzes mc truth
process.load( 'L1Trigger.TrackTrigger.AnalyzerMC_cff' )
# DTC emulation
process.load('L1Trigger.TrackerDTC.DTC_cff')

# load code that analyzes DTCStubs
process.load('L1Trigger.TrackerDTC.Analyzer_cff')

# modify default cuts
#process.TrackTriggerSetup.FrontEnd.BendCut = 5.0
#process.TrackTriggerSetup.Hybrid.MinPt = 1.0

process.dtc = cms.Path(process.StubAssociator + process.AnalyzerMC + process.ProducerDTC + process.AnalyzerDTC)

############################################################
# L1 tracking
############################################################

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")

# HYBRID: prompt tracking
if (L1TRKALGO == 'HYBRID'):
    process.load( 'L1Trigger.TrackFindingTracklet.Analyzer_cff' )
    process.TTTracksEmulation = cms.Path(process.L1THybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1THybridTracksWithAssociators + process.AnalyzerTracklet)
    NHELIXPAR = 4
    L1TRK_NAME  = "l1tTTTracksFromTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigis"

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_DISPLACED'):
    NHELIXPAR = 5
    L1TRK_NAME  = "l1tTTTracksFromExtendedTrackletEmulation"
    L1TRK_LABEL = "Level1TTTracks"
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigisExtended"
    process.load( 'L1Trigger.TrackFindingTracklet.Analyzer_cff' )
    process.AnalyzerTracklet.InputTag = cms.InputTag(L1TRK_NAME, L1TRK_LABEL)
    process.TTTracksEmulation = cms.Path(process.L1TExtendedHybridTracks)
    process.TTTracksEmulationWithTruth = cms.Path(process.L1TExtendedHybridTracksWithAssociators + process.AnalyzerTracklet)

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
    # Optionally include code producing performance plots & end-of-job summary.
    if (TRAINING_RUN == True):
        process.TTTracksEmulationWithTruth = cms.Path(process.HybridNewKF + process.TrackTriggerAssociatorTracks + process.AnalyzerTracklet + process.AnalyzerTM + process.AnalyzerDR + process.AnalyzerKF + process.AnalyzerTQ + process.AnalyzerTFP )
    else:
        process.TTTracksEmulationWithTruth = cms.Path(process.HybridNewKF + process.TrackTriggerAssociatorTracks + process.AnalyzerTracklet + process.AnalyzerTM + process.AnalyzerDR + process.AnalyzerKF + process.AnalyzerTFP )
    from L1Trigger.TrackFindingTracklet.Customize_cff import *
    if (L1TRKALGO == 'HYBRID_NEWKF'):
        fwConfig( process )
        # cheats to get good performance
        process.TrackFindingTrackletSetup.DR.UseTTStubs = True
    if (L1TRKALGO == 'HYBRID_REDUCED'):
        reducedConfig( process )
    # Needed by L1TrackNtupleMaker
    process.HitPatternHelperSetup.useNewKF = True

# HYBRID: extended tracking followd by 5 param fit sim
elif (L1TRKALGO == 'HYBRID_DISPLACED_SIM'):
    process.load( 'L1Trigger.TrackFindingTracklet.Producer_cff' )
    process.load( 'L1Trigger.TrackFindingTracklet.Analyzer_cff' )
    process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
    from L1Trigger.TrackFindingTracklet.Customize_cff import *
    NHELIXPAR = 5
    TRACKLET_NAME  = "l1tTTTracksFromExtendedTrackletEmulation"
    TRACKLET_LABEL = "Level1TTTracks"
    L1TRK_NAME  = "ProducerSim"
    L1TRK_LABEL = process.TrackFindingTrackletProducer_params.BranchTTTracks.value()
    L1TRUTH_NAME = "TTTrackAssociatorFromPixelDigisExtended"
    process.TTTrackAssociatorFromPixelDigisExtended.TTTracks = cms.VInputTag( cms.InputTag(L1TRK_NAME, L1TRK_LABEL) )
    process.AnalyzerTracklet.InputTag = cms.InputTag(TRACKLET_NAME, TRACKLET_LABEL)
    process.StubAssociator.MaxZ0 = 30.
    process.StubAssociator.MaxD0 = 10.
    process.StubAssociator.MaxVertR = 10.
    process.StubAssociator.MaxVertZ = 60.
    from L1Trigger.TrackFindingTracklet.Customize_cff import *
    simConfig( process )
    process.Sim = cms.Sequence(process.L1TExtendedHybridTracks + process.ProducerSim)
    process.TTTracksEmulationWithTruth = cms.Path(process.Sim + process.L1TExtendedHybridTracksWithAssociators + process.StubAssociator + process.AnalyzerTracklet + process.AnalyzerSim)
    process.TTTracksEmulation = cms.Path(process.Sim)

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



