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
process.load( 'Configuration.Geometry.GeometryExtended2026D76Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtended2026D76_cff' )
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
process.tracklet = cms.Sequence( process.L1HybridTracks + process.TrackFindingTrackletAnalyzerTracklet )
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
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/05f802b7-b0b3-4cca-8b70-754682c3bb4c.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/0b69ed0a-66e9-403a-88f0-fb3115615461.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/0f4dea68-7574-43bb-97c3-5382d68a2704.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/156b3ca6-c74a-4f46-ae5e-03d9b01acd4c.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/16727f1d-2922-4e0a-8239-82e1ffecd43b.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/1af620bf-1f6d-4d5a-8170-4135ac798581.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/1dc513d9-75fc-44c0-b8e0-e2925323416b.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/2010f402-2133-4c3a-851b-1ae68fe23eb3.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/228dfbba-3d5c-42b9-b827-cfa8f11a2f38.root',
  #'/store/relval/CMSSW_11_3_0_pre6/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/10000/27d006b1-d023-4775-8430-382e6962149c.root'
  #'/store/relval/CMSSW_11_3_0_pre6/RelValDisplacedMuPt2To100Dxy100/GEN-SIM-DIGI-RAW/113X_mcRun4_realistic_v6_2026D76noPU-v1/00000/011da61a-9524-4a96-b91f-03e8690af3bd.root'
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/00026541-6200-4eed-b6f8-d3a1fd720e9c.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/013d0125-8f6e-496b-8335-614398c9210d.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/058bd134-86de-47e1-bcde-379ed9b79e1b.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/0915d66c-cbd4-4ef6-9971-7dd59e198b56.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/09823c8d-e443-4066-8347-8c704929cb2b.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/0c39a1aa-93ee-41c1-8543-6d90c09114a7.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/0fcdcc53-fb9f-4f0b-8529-a4d60d914c14.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/16760a5c-9cd2-41c3-82e5-399bb962d537.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/1752640f-2001-4d14-9276-063ec07cea92.root',
  '/store/relval/CMSSW_11_3_0_pre6/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_113X_mcRun4_realistic_v6_2026D76PU200-v1/00000/180712c9-31a5-4f2a-bf92-a7fbee4dabad.root'
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