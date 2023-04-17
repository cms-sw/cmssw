################################################################################################
# This script runs DTC + prompt tracklet + KF interface + new KF emulator with analyzer for each step
# allowing to identify problems quickly during developement.
# This script is a specialized and light-weight version of L1TrackNtupleMaker_cfg.py
# To run execute do
# cmsRun L1Trigger/TrackFindingTracklet/test/HybridTracksNewKF_cfg.py
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process( "Demo" )
process.load( 'FWCore.MessageService.MessageLogger_cfi' )
process.load( 'Configuration.EventContent.EventContent_cff' )
process.load( 'Configuration.Geometry.GeometryExtended2026D88Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtended2026D88_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'L1Trigger.TrackTrigger.TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# load code that associates stubs with mctruth
process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.ProducerED_cff' )
# load code that analyzes DTCStubs
process.load( 'L1Trigger.TrackerDTC.Analyzer_cff' )
# L1 tracking => hybrid emulation 
process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
from L1Trigger.TrackFindingTracklet.Customize_cff import *
fwConfig( process )
#--- Load code that analyzes hybrid emulation 
process.load( 'L1Trigger.TrackFindingTracklet.Analyzer_cff' )
# load code that fits hybrid tracks
process.load( 'L1Trigger.TrackFindingTracklet.Producer_cff' )

# load and configure TrackTriggerAssociation
process.load( 'SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff' )
process.TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag( cms.InputTag(
  process.TrackFindingTrackletProducer_params.LabelTT.value(),
  process.TrackFindingTrackletProducer_params.BranchAcceptedTracks.value()
) )

# build schedule
process.mc = cms.Sequence( process.StubAssociator )
process.dtc = cms.Sequence( process.TrackerDTCProducer + process.TrackerDTCAnalyzer )
process.tracklet = cms.Sequence( process.L1THybridTracks + process.TrackFindingTrackletAnalyzerTracklet )
process.TBout = cms.Sequence( process.TrackFindingTrackletProducerTBout + process.TrackFindingTrackletAnalyzerTBout )
process.interIn = cms.Sequence( process.TrackFindingTrackletProducerKFin + process.TrackFindingTrackletAnalyzerKFin )
process.kf = cms.Sequence( process.TrackFindingTrackletProducerKF + process.TrackFindingTrackletAnalyzerKF )
process.TTTracks = cms.Sequence( process.TrackFindingTrackletProducerTT + process.TrackFindingTrackletProducerAS + process.TrackTriggerAssociatorTracks )
process.interOut = cms.Sequence( process.TrackFindingTrackletProducerKFout + process.TrackFindingTrackletAnalyzerKFout )
process.tt = cms.Path( process.mc + process.dtc + process.tracklet + process.TBout + process.interIn + process.kf + process.TTTracks + process.interOut )
process.schedule = cms.Schedule( process.tt )

# create options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing( 'analysis' )
# specify input MC
#from MCsamples.Scripts.getCMSdata_cfi import *
#from MCsamples.Scripts.getCMSlocaldata_cfi import *
#from MCsamples.RelVal_1130_D76.PU200_TTbar_14TeV_cfi import *
#inputMC = getCMSdataFromCards()
inputMC = [
'/store/mc/CMSSW_12_6_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_125X_mcRun4_realistic_v5_2026D88PU200RV183v2-v1/30000/0959f326-3f52-48d8-9fcf-65fc41de4e27.root'
]
options.register( 'inputMC', inputMC, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed" )
# specify number of events to process.
options.register( 'Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of Events to analyze" )
options.parseArguments()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )
process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring( options.inputMC ),
  #skipEvents = cms.untracked.uint32( 250 ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)
process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
process.MessageLogger.cerr.enableStatistics = False
process.TFileService = cms.Service( "TFileService", fileName = cms.string( "Hist.root" ) )

if ( False ):
  process.out = cms.OutputModule (
    "PoolOutputModule",
    fileName = cms.untracked.string("L1Tracks.root"),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_TTTrack*_*_*', 'keep *_TTStub*_*_*' )
  )
  process.FEVToutput_step = cms.EndPath( process.out )
  process.schedule.append( process.FEVToutput_step )
