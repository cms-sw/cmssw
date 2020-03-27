################################################################################################
# To run execute do
# cmsRun tmtt_tf_analysis_cfg.py Events=50 inputMC=Samples/Muons/PU0.txt histFile=outputHistFile.root trkFitAlgo=All
# where the arguments take default values if you don't specify them. You can change defaults below.
# (To run a subset of track fitters, specify, for example: trackFitAlgo=SimpleLR,Tracklet ).
#################################################################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Demo")

process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.load("FWCore.MessageLogger.MessageLogger_cfi")

options = VarParsing.VarParsing ('analysis')

#--- Specify input MC
# options.register('inputMC', 'MCsamples/932/RelVal/eclementMC/TTbar/PU200.txt',
options.register('inputMC', 'MCsamples/937/RelVal/TTbar/PU200.txt',

# Fastest to use a local copy ...
#options.register('inputMC', 'MCsamples/932/RelVal/eclementMC/TTbar/localRAL/PU200.txt', 

VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed")

#--- Specify number of events to process.
options.register('Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,"Number of Events to analyze")

#--- Specify name of output histogram file.
options.register('histFile','Hist.root',VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,"Name of output histogram file")

#--- Specify which track finding algorithms
#--- Options are KF4ParamsComb, SimpleLR, Tracklet, All
#--- Can provide comma separated list
options.register('trkFitAlgo','KF4ParamsComb',VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,"Name of track helix fit algorithm")

#--- Specify if stubs need to be produced i.e. they are not available in the input file
options.register('makeStubs',0,VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.int,"Make stubs, and truth association, on the fly")

options.parseArguments()

options.trkFitAlgo = options.trkFitAlgo.split(',')
if 'All' in options.trkFitAlgo:
  options.trkFitAlgo = ['KF4ParamsComb', 'KF5ParamsComb', 'Tracklet', 'SimpleLR']

#--- input and output

list = FileUtils.loadListFromFile(options.inputMC)
readFiles = cms.untracked.vstring(*list)
# readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

outputFileName = options.histFile

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(outputFileName)
)

process.source = cms.Source ("PoolSource",
                            fileNames = readFiles,
                            secondaryFileNames = secFiles,
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            # Following needed to read CMSSW 9 datasets with CMSSW 10
                            inputCommands = cms.untracked.vstring(
                              'keep *_*_*_*',
                              'drop l1tEMTFHit2016*_*_*_*',
                              'drop l1tEMTFTrack2016*_*_*_*'
                            )
                            )

process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

#--- Load code that produces our L1 tracks and makes corresponding histograms.
process.load('L1Trigger.TrackFindingTMTT.TMTrackProducer_cff')

#--- Alternative cfg including improvements not yet in the firmware. Aimed at L1 trigger studies.
# process.load('L1Trigger.TrackFindingTMTT.TMTrackProducer_Ultimate_cff')
#
#--- Optionally override default configuration parameters here (example given of how).
#process.TMTrackProducer.TrackFitSettings.TrackFitters = cms.vstring()


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
                                       TP_maxEta = cms.double(2.4),      # only save TPs with |eta| < X
                                       TP_maxZ0 = cms.double(30.0),      # only save TPs with |z0| < X cm
                                       L1TrackInputTag = cms.InputTag("TMTrackProducer", "TML1TracksKF4ParamsComb"),               ## TTTrack input
                                       MCTruthTrackInputTag = cms.InputTag("TTAssociatorTMTT", "TML1TracksKF4ParamsComb"), ## MCTruth input 
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

#--- Run TMTT track production
#--- Also run the ntuple production for each tracking algorithm
process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')

process.TMTrackProducer.TrackFitSettings.TrackFitters = cms.vstring()
#--- TMTT tracks : KF4ParamsComb
if 'KF4ParamsComb' in options.trkFitAlgo:
  process.TMTrackProducer.TrackFitSettings.TrackFitters.append("KF4ParamsComb")

  process.TTAssociatorTMTTKF4ParamsComb = process.TTTrackAssociatorFromPixelDigis.clone(
          TTTracks = cms.VInputTag(cms.InputTag("TMTrackProducer", 'TML1TracksKF4ParamsComb'))
      )
  process.L1TrackNtuple_TMTT_KF4ParamsComb = process.L1TrackNtuple.clone(
          L1TrackInputTag = cms.InputTag("TMTrackProducer", 'TML1TracksKF4ParamsComb'),
          MCTruthTrackInputTag = cms.InputTag("TTAssociatorTMTTKF4ParamsComb", 'TML1TracksKF4ParamsComb'),
        )
  process.p_TMTT_KF4ParamsComb = cms.Path( process.TMTrackProducer*process.TTAssociatorTMTTKF4ParamsComb*process.L1TrackNtuple_TMTT_KF4ParamsComb )

if 'KF5ParamsComb' in options.trkFitAlgo:
  process.TMTrackProducer.TrackFitSettings.TrackFitters.append("KF5ParamsComb")

  process.TTAssociatorTMTTKF5ParamsComb = process.TTTrackAssociatorFromPixelDigis.clone(
          TTTracks = cms.VInputTag(cms.InputTag("TMTrackProducer", 'TML1TracksKF5ParamsComb'))
      )
  process.L1TrackNtuple_TMTT_KF5ParamsComb = process.L1TrackNtuple.clone(
          L1TrackInputTag = cms.InputTag("TMTrackProducer", 'TML1TracksKF5ParamsComb'),
          MCTruthTrackInputTag = cms.InputTag("TTAssociatorTMTTKF5ParamsComb", 'TML1TracksKF5ParamsComb'),
          L1Tk_nPar = cms.int32(5),
        )
  process.p_TMTT_KF5ParamsComb = cms.Path( process.TMTrackProducer*process.TTAssociatorTMTTKF5ParamsComb*process.L1TrackNtuple_TMTT_KF5ParamsComb )


#--- TMTT tracks : SimpleLR
if 'SimpleLR' in options.trkFitAlgo:
  process.TMTrackProducer.TrackFitSettings.TrackFitters.append("SimpleLR")

  process.TTAssociatorTMTTSimpleLR = process.TTTrackAssociatorFromPixelDigis.clone(
          TTTracks = cms.VInputTag(cms.InputTag("TMTrackProducer", 'TML1TracksSimpleLR'))
      )
  process.L1TrackNtuple_TMTT_SimpleLR = process.L1TrackNtuple.clone(
          L1TrackInputTag = cms.InputTag("TMTrackProducer", 'TML1TracksSimpleLR'),
          MCTruthTrackInputTag = cms.InputTag("TTAssociatorTMTTSimpleLR", 'TML1TracksSimpleLR'),
        )
  process.p_TMTT_SimpleLR = cms.Path( process.TMTrackProducer*process.TTAssociatorTMTTSimpleLR*process.L1TrackNtuple_TMTT_SimpleLR )


#--- Tracklet tracks
if 'Tracklet' in options.trkFitAlgo:
  process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
  process.TTAssociatorTracklet = process.TTTrackAssociatorFromPixelDigis.clone(
          TTTracks = cms.VInputTag(cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"))
      )
  process.L1TrackNtuple_Tracklet = process.L1TrackNtuple.clone(
          L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", 'Level1TTTracks'),
          MCTruthTrackInputTag = cms.InputTag("TTAssociatorTracklet", 'Level1TTTracks'),
        )
  process.TrackletPath = cms.Path(process.offlineBeamSpot*process.TTTracksFromTracklet*process.TTAssociatorTracklet*process.L1TrackNtuple_Tracklet)



# Optionally reproduce the stubs
if options.makeStubs == 1:
  process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
  process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
  process.TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")
  process.p = cms.Path(process.TrackTriggerClustersStubs * process.TrackTriggerAssociatorClustersStubs * process.TMTrackProducer)


from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
