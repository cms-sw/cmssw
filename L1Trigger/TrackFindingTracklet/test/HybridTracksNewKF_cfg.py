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
process.load( 'Configuration.Geometry.GeometryExtendedRun4D98Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtendedRun4D98_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'L1Trigger.TrackTrigger.TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')

# load code that associates stubs with mctruth
process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.DTC_cff' )
# load code that analyzes DTCStubs
process.load( 'L1Trigger.TrackerDTC.Analyzer_cff' )
# L1 tracking => hybrid emulation 
process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
#--- Load code that analyzes hybrid emulation 
process.load( 'L1Trigger.TrackFindingTracklet.Analyzer_cff' )
# load code that fits hybrid tracks
process.load( 'L1Trigger.TrackFindingTracklet.Producer_cff' )
from L1Trigger.TrackFindingTracklet.Customize_cff import *
fwConfig( process )
oldKFConfig( process )
process.l1tTTTracksFromTrackletEmulation.readMoreMcTruth = False

# build schedule
process.mc       = cms.Sequence( process.StubAssociator                             )
process.dtc      = cms.Sequence( process.ProducerDTC     + process.AnalyzerDTC      )
process.tracklet = cms.Sequence( process.L1THybridTracks + process.AnalyzerTracklet )
process.tm       = cms.Sequence( process.ProducerTM      + process.AnalyzerTM       )
process.dr       = cms.Sequence( process.ProducerDR      + process.AnalyzerDR       )
process.kf       = cms.Sequence( process.ProducerKF      + process.AnalyzerKF       )
process.tq       = cms.Sequence( process.ProducerTQ      + process.AnalyzerTQ       )
process.tfp      = cms.Sequence( process.ProducerTFP     + process.AnalyzerTFP      )
process.tt       = cms.Path( process.mc + process.dtc + process.tracklet + process.tm + process.dr + process.kf + process.tq + process.tfp )
process.schedule = cms.Schedule( process.tt )

# create options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing( 'analysis' )
# specify input MC
#from MCsamples.Scripts.getCMSdata_cfi import *
#from MCsamples.Scripts.getCMSlocaldata_cfi import *
#from MCsamples.RelVal_1260_D88.PU200_TTbar_14TeV_cfi import *
#inputMC = getCMSdataFromCards()
Samples = [
  '/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0b2b0b0b-f312-48a8-9d46-ccbadc69bbfd.root'
]
options.register( 'inputMC', Samples, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed" )
# specify number of events to process.
options.register( 'Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of Events to analyze" )
options.parseArguments()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )
process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring( options.inputMC ),
  #skipEvents = cms.untracked.uint32( 3537 ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)
process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
process.MessageLogger.cerr.enableStatistics = False
process.MessageLogger.L1track = dict(limit = -1)
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
