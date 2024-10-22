import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('inputTag',
                 'LayerRot_9p43e-6',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "input tag")
options.register('inputFile',
                 '/store/relval/CMSSW_14_0_0_pre1/RelValZMM_14/GEN-SIM-RECO/133X_mcRun3_2023_realistic_v3-v1/2590000/586487a4-71be-4b23-b5a4-5662fab803c9.root',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "input file")
options.register('isAlCaReco',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "is alcareco input file?")
options.register('isUnitTest',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "is this configuration run in unit test?")
options.parseArguments()

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("TrackingResolution", Run3)

#####################################################################
# import of standard configurations
#####################################################################
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = (100 if options.isUnitTest else 100000)
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#####################################################################
## BeamSpot from database (i.e. GlobalTag), needed for Refitter
#####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

#####################################################################
# Load and Configure Measurement Tracker Event
#####################################################################
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
if(options.isAlCaReco):
    # customize MeasurementTrackerEvent for ALCARECO
    process.MeasurementTrackerEvent.pixelClusterProducer = "ALCARECOTkAlDiMuon"
    process.MeasurementTrackerEvent.stripClusterProducer = "ALCARECOTkAlDiMuon"
    process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
    process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10 if options.isUnitTest else -1)
)

#####################################################################
# Input source
#####################################################################
#filelist = FileUtils.loadListFromFile("listOfFiles_idealMC_GEN-SIM-RECO.txt")
#filelist = FileUtils.loadListFromFile("listOfFiles_idealMC_TkAlDiMuonAndVertex.txt")
#readFiles = cms.untracked.vstring( *filelist)

readFiles = cms.untracked.vstring(options.inputFile)
process.source = cms.Source("PoolSource",fileNames = readFiles)

process.options = cms.untracked.PSet()

#####################################################################
# Output 
#####################################################################
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step1_DQM_'+options.inputTag+'_'+('fromALCA' if options.isAlCaReco else 'fromRECO' )+'.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

#####################################################################
# Other statements
#####################################################################
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag,"133X_mcRun3_2023_realistic_v3", '')
process.GlobalTag = GlobalTag(process.GlobalTag, "125X_mcRun3_2022_design_v6", '')
process.GlobalTag.toGet = cms.VPSet(cms.PSet(connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
                                             record = cms.string('TrackerAlignmentRcd'),
                                             tag = cms.string(options.inputTag)))

#####################################################################
# The DQM analysis sequence
#####################################################################
process.load("DQM.TrackingMonitorSource.shortTrackResolution_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.LongTracksRefit = process.TrackRefitter.clone(
    src = 'SingleLongTrackProducer',
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate",
    NavigationSchool = ''
)

process.ShortTrackCandidates3.src = cms.InputTag("LongTracksRefit")
process.ShortTrackCandidates4.src = cms.InputTag("LongTracksRefit")
process.ShortTrackCandidates5.src = cms.InputTag("LongTracksRefit") 
process.ShortTrackCandidates6.src = cms.InputTag("LongTracksRefit") 
process.ShortTrackCandidates7.src = cms.InputTag("LongTracksRefit") 
process.ShortTrackCandidates8.src = cms.InputTag("LongTracksRefit") 

process.SingleLongTrackProducer.matchMuons = cms.InputTag("muons")
if(options.isAlCaReco):
    process.SingleLongTrackProducer.requiredDr = cms.double(-9999.) # do not require any matchings
    process.SingleLongTrackProducer.allTracks = cms.InputTag("ALCARECOTkAlDiMuon")

#####################################################################
# Path
#####################################################################
process.analysis_step = cms.Path(process.offlineBeamSpot *
                                 process.MeasurementTrackerEvent *
                                 process.SingleLongTrackProducer *
                                 process.LongTracksRefit *
                                 process.ShortTrackCandidates3 *
                                 process.ShortTrackCandidates4 *
                                 process.ShortTrackCandidates5 *
                                 process.ShortTrackCandidates6 *
                                 process.ShortTrackCandidates7 *
                                 process.ShortTrackCandidates8 *
                                 process.RefittedShortTracks3 *
                                 process.RefittedShortTracks4 *
                                 process.RefittedShortTracks5 *
                                 process.RefittedShortTracks6 *
                                 process.RefittedShortTracks7 *
                                 process.RefittedShortTracks8 *
                                 process.trackingResolution)

#####################################################################
# Path and EndPath definitions
#####################################################################
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.schedule = cms.Schedule(process.analysis_step, process.endjob_step, process.DQMoutput_step)

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8
