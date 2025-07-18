import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing

# PART 1 : PARSE ARGUMENTS

options = VarParsing.VarParsing ('analysis')
options.register('debug',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Print out additional debugging information")
options.register ('format',
                  'EMP', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "File format (APx, EMP or X2O)")
options.register('threads',
                 1, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of threads to run")
options.register('streams',
                 0, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of streams to run")
options.register ('tracks',
                  'donotload', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Whether to load tracks from buffers and how to treat them in the processing chain ('donotload', 'load', 'overwrite')")
options.register ('vertices',
                  'donotload', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Whether to load vertices from buffers and how to treat them in the processing chain ('donotload', 'load', 'overwrite')")
options.register ('readerformat',
                  'EMPv2', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "File format of loaded tracks and vertices (APx, EMPv2)")
options.parseArguments()

inputFiles = []
inputBuffers = []
inputTrackBuffers = []
for filePath in options.inputFiles:
    if filePath.endswith(".root"):
        inputFiles.append(filePath)
    elif filePath.endswith("_cff.py"):
        filePath = filePath.replace("/python/","/")
        filePath = filePath.replace("/", ".")
        inputFilesImport = getattr(__import__(filePath.strip(".py"),fromlist=["readFiles"]),"readFiles")
        inputFiles.extend( inputFilesImport )
        if options.vertices in ['load', 'overwrite']:
            inputBuffersImport = getattr(__import__(filePath.strip(".py"),fromlist=["correlator_source"]),"correlator_source").fileNames
            inputBuffers.extend( inputBuffersImport )
        if options.tracks in ['load', 'overwrite']:
            inputTrackBuffersImport = getattr(__import__(filePath.strip(".py"),fromlist=["track_source"]),"track_source").fileNames
            inputTrackBuffers.extend( inputTrackBuffersImport )
    else:
        inputFiles += FileUtils.loadListFromFile(filePath)

# PART 2: SETUP MAIN CMSSW PROCESS 


process = cms.Process("GTTFileWriter")

process.load('Configuration.Geometry.GeometryExtendedRun4D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D88_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inputFiles),
    inputCommands = cms.untracked.vstring("keep *", "drop l1tTkPrimaryVertexs_L1TkPrimaryVertex__*")
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.threads),
    numberOfStreams = cms.untracked.uint32(options.streams if options.streams>0 else 0)
)

process.load('L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackSelectionProducer_cfi')
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackVertexAssociationProducer_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackJetsEmulation_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackerEmuHTMiss_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackerEmuEtMiss_cfi')
process.load('L1Trigger.DemonstratorTools.l1tGTTFileWriter_cfi')
process.load('L1Trigger.DemonstratorTools.l1tGTTFileReader_cfi')

process.l1tGTTFileReader.processOutputToCorrelator = cms.bool((options.vertices in ['load', 'overwrite']))
process.l1tGTTFileReader.processInputTracks = cms.bool((options.tracks in ['load', 'overwrite']))
process.l1tGTTFileReader.processOutputToGlobalTrigger = cms.bool(False) #NotImplemented
process.l1tGTTFileReader.filesOutputToCorrelator = inputBuffers if (options.vertices in ['load', 'overwrite']) else cms.vstring("L1GTTOutputToCorrelatorFile_0.txt")
process.l1tGTTFileReader.filesInputTracks = inputTrackBuffers if (options.tracks in ['load', 'overwrite']) else cms.vstring("L1GTTInputFile_0.txt")
process.l1tGTTFileReader.filesOutputToGlobalTrigger = cms.vstring("L1GTTOutputToGlobalTriggerFile_0.txt")
process.l1tGTTFileReader.format = cms.untracked.string(options.readerformat)

process.l1tGTTInputProducer.debug = cms.int32(options.debug)
if (options.tracks in ['overwrite']):
    process.l1tGTTInputProducer.l1TracksInputTag = cms.InputTag("l1tGTTFileReader", "Level1TTTracks")
    process.l1tGTTInputProducer.setTrackWordBits = cms.bool(False)

process.l1tTrackSelectionProducer.processSimulatedTracks = cms.bool(False)
process.l1tVertexFinderEmulator.VertexReconstruction.VxMinTrackPt = cms.double(0.0)
process.l1tVertexFinderEmulator.debug = options.debug
process.l1tTrackVertexAssociationProducer.processSimulatedTracks = cms.bool(False)

process.l1tTrackSelectionProducerForEtMiss.processSimulatedTracks = cms.bool(False)
process.l1tTrackVertexAssociationProducerForEtMiss.processSimulatedTracks = cms.bool(False)
process.l1tTrackerEmuEtMiss.debug = options.debug

process.l1tTrackSelectionProducerForJets.processSimulatedTracks = cms.bool(False)
process.l1tTrackSelectionProducerForJets.cutSet = cms.PSet(
    ptMin = cms.double(2.0), # pt must be greater than this value, [GeV]
    absEtaMax = cms.double(2.4), # absolute value of eta must be less than this value
    absZ0Max = cms.double(15.0), # z0 must be less than this value, [cm]
    nStubsMin = cms.int32(4), # number of stubs must be greater than or equal to this value
    nPSStubsMin = cms.int32(0), # the number of stubs in the PS Modules must be greater than or equal to this value
    
    promptMVAMin = cms.double(-1.0), # MVA must be greater than this value
    reducedBendChi2Max = cms.double(2.25), # bend chi2 must be less than this value
    reducedChi2RZMax = cms.double(5.0), # chi2rz/dof must be less than this value
    reducedChi2RPhiMax = cms.double(20.0), # chi2rphi/dof must be less than this value
)
process.l1tTrackVertexAssociationProducerForJets.processSimulatedTracks = cms.bool(False)
process.l1tTrackVertexAssociationProducerForJets.cutSet = cms.PSet(
    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
    deltaZMax = cms.vdouble(0.37, 0.50, 0.60, 0.75, 1.00, 1.60), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
)
process.l1tTrackerEmuHTMiss.debug = (options.debug > 0)

#Disable internal track selection
process.l1tTrackJetsEmulation.trk_zMax = cms.double(20.46912512)    # maximum track z from TrackWord

if options.debug:
    process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(1000000000)
    process.MessageLogger.suppressInfo = cms.untracked.vstring('CondDBESSource', 'PoolDBESSource')
    process.MessageLogger.cerr.CondDBESSource = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )

process.l1tGTTFileWriter.format = cms.untracked.string(options.format) #FIXME Put all this into the default GTTFileWriter
if options.tracks == 'overwrite':
    process.l1tGTTFileWriter.tracks = cms.untracked.InputTag("l1tGTTFileReader", "Level1TTTracks")
else:
    process.l1tGTTFileWriter.tracks = cms.untracked.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks")
process.l1tGTTFileWriter.convertedTracks = cms.untracked.InputTag("l1tGTTInputProducer", "Level1TTTracksConverted")
process.l1tGTTFileWriter.selectedTracks = cms.untracked.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedEmulation")
if options.vertices == 'overwrite':
    process.l1tGTTFileWriter.vertices = cms.untracked.InputTag("l1tGTTFileReader", "L1VerticesFirmware")
else:
    process.l1tGTTFileWriter.vertices = cms.untracked.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation")
process.l1tGTTFileWriter.vertexAssociatedTracks = cms.untracked.InputTag("l1tTrackVertexAssociationProducer", "Level1TTTracksSelectedAssociatedEmulation")
process.l1tGTTFileWriter.jets = cms.untracked.InputTag("l1tTrackJetsEmulation","L1TrackJets")
process.l1tGTTFileWriter.htmiss = cms.untracked.InputTag("l1tTrackerEmuHTMiss", "L1TrackerEmuHTMiss")
process.l1tGTTFileWriter.etmiss = cms.untracked.InputTag("l1tTrackerEmuEtMiss", "L1TrackerEmuEtMiss")
process.l1tGTTFileWriter.outputCorrelatorFilename = cms.untracked.string("L1GTTOutputToCorrelatorFile")
process.l1tGTTFileWriter.outputGlobalTriggerFilename = cms.untracked.string("L1GTTOutputToGlobalTriggerFile")
process.l1tGTTFileWriter.selectedTracksFilename = cms.untracked.string("L1GTTSelectedTracksFile")
process.l1tGTTFileWriter.vertexAssociatedTracksFilename = cms.untracked.string("L1GTTVertexAssociatedTracksFile")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

if options.tracks in ['load', 'overwrite'] or options.vertices in ['load', 'overwrite']:
    process.p = cms.Path(process.l1tGTTFileReader * process.l1tGTTFileWriter)
else:
    process.p = cms.Path(process.l1tGTTFileWriter)
process.p.associate(cms.Task(process.l1tGTTInputProducer, 
                             process.l1tTrackSelectionProducer,
                             process.l1tVertexFinderEmulator, 
                             process.l1tTrackVertexAssociationProducer,
                             process.l1tTrackSelectionProducerForJets,
                             process.l1tTrackVertexAssociationProducerForJets,
                             process.l1tTrackJetsEmulation, 
                             process.l1tTrackerEmuHTMiss, 
                             process.l1tTrackSelectionProducerForEtMiss,
                             process.l1tTrackVertexAssociationProducerForEtMiss,
                             process.l1tTrackerEmuEtMiss,
                         )
                )
