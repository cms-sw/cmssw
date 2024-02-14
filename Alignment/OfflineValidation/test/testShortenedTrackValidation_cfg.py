import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('scenario', 
                 '0',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "Name of input misalignment scenario")
options.parseArguments()

valid_scenarios = ['-10e-6','-8e-6','-6e-6','-4e-6','-2e-6','0','2e-6','4e-6','6e-6','8e-6','10e-6']

if options.scenario not in valid_scenarios:
    print("Error: Invalid scenario specified. Please choose from the following list: ")
    print(valid_scenarios)
    exit(1)

process = cms.Process("TrackingResolution")

#####################################################################
# import of standard configurations
#####################################################################
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100000
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#####################################################################
## BeamSpot from database (i.e. GlobalTag), needed for Refitter
#####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

#####################################################################
# Load and Configure Measurement Tracker Event
#####################################################################
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
process.MeasurementTrackerEvent.pixelClusterProducer = "ALCARECOTkAlDiMuon"
process.MeasurementTrackerEvent.stripClusterProducer = "ALCARECOTkAlDiMuon"
process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)

#####################################################################
# Input source
#####################################################################
# filelist = FileUtils.loadListFromFile("listOfFiles_idealMC_TkAlDiMuonAndVertex.txt")
# readFiles = cms.untracked.vstring( *filelist)
# events taken from /DYJetsToMuMu_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter23Reco-TkAlDiMuonAndVertex-TRKDesignNoPU_AlcaRecoTRKMu_designGaussSigmaZ4cm_125X_mcRun3_2022_design_v6-v1/ALCARECO
readFiles = cms.untracked.vstring('/store/mc/Run3Winter23Reco/DYJetsToMuMu_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/ALCARECO/TkAlDiMuonAndVertex-TRKDesignNoPU_AlcaRecoTRKMu_designGaussSigmaZ4cm_125X_mcRun3_2022_design_v6-v1/60000/d3af17a5-2409-4551-9c3d-00deb2f3f64f.root')
process.source = cms.Source("PoolSource",fileNames = readFiles)

process.options = cms.untracked.PSet()

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("shortenedTrackResolution_LayerRotation_"+options.scenario+".root"))

#####################################################################
# Other statements
#####################################################################
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "125X_mcRun3_2022_design_v6", '')
if (options.scenario=='null'):
    print("null scenario, do nothing")
    pass
else:
    process.GlobalTag.toGet = cms.VPSet(cms.PSet(connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"),
                                                 record = cms.string('TrackerAlignmentRcd'),
                                                 tag = cms.string("LayerRotation_"+options.scenario)))

#####################################################################
# The DQM analysis sequence
#####################################################################
process.load("DQM.TrackingMonitorSource.shortTrackResolution_cff")

#####################################################################
# The changes to cope with ALCARECO data format
#####################################################################
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

process.SingleLongTrackProducer.requiredDr = cms.double(-9999.) # do not require any matchings
process.SingleLongTrackProducer.matchMuons = cms.InputTag("muons") # for ALCA irrelevant (see above)
process.SingleLongTrackProducer.allTracks = cms.InputTag("ALCARECOTkAlDiMuon")

#####################################################################
# The Analysis module
#####################################################################
from Alignment.OfflineValidation.shortenedTrackValidation_cfi import shortenedTrackValidation as _shortenedTrackValidation
process.ShortenedTrackValidation = _shortenedTrackValidation.clone(folderName           = "ShortTrackResolution",
                                                                   hitsRemainInput      = ["3","4","5","6","7","8"],
                                                                   minTracksEtaInput    = 0.0,
                                                                   maxTracksEtaInput    = 2.2,
                                                                   minTracksPtInput     = 15.0,
                                                                   maxTracksPtInput     = 99999.9,
                                                                   maxDrInput           = 0.01,
                                                                   tracksInputTag       = "SingleLongTrackProducer",
                                                                   tracksRerecoInputTag = ["RefittedShortTracks3",
                                                                                           "RefittedShortTracks4",
                                                                                           "RefittedShortTracks5",
                                                                                           "RefittedShortTracks6",
                                                                                           "RefittedShortTracks7",
                                                                                           "RefittedShortTracks8"])

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
                                 process.ShortenedTrackValidation)

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8
