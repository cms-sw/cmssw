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
options.parseArguments()

inputFiles = []
for filePath in options.inputFiles:
    if filePath.endswith(".root"):
        inputFiles.append(filePath)
    elif filePath.endswith("_cff.py"):
        filePath = filePath.replace("/python/","/")
        filePath = filePath.replace("/", ".")
        inputFilesImport = getattr(__import__(filePath.strip(".py"),fromlist=["readFiles"]),"readFiles")
        inputFiles.extend( inputFilesImport )
    else:
        inputFiles += FileUtils.loadListFromFile(filePath)

# PART 2: SETUP MAIN CMSSW PROCESS 

process = cms.Process("GTTFileWriter")

process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
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
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
process.load("L1Trigger.L1TTrackMatch.l1tTrackSelectionProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackJetsEmulation_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackerEmuHTMiss_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackerEmuEtMiss_cfi")
process.load('L1Trigger.DemonstratorTools.GTTFileWriter_cff')

process.l1tGTTInputProducer.debug = cms.int32(options.debug)
process.l1tVertexProducer.l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted")
process.l1tVertexProducer.VertexReconstruction.Algorithm = cms.string("fastHistoEmulation")
process.l1tVertexProducer.VertexReconstruction.VxMinTrackPt = cms.double(0.0)
process.l1tVertexProducer.debug = options.debug
process.l1tTrackSelectionProducer.processSimulatedTracks = cms.bool(False)
process.l1tTrackSelectionProducer.l1VerticesEmulationInputTag = cms.InputTag("l1tVertexProducer", "l1verticesEmulation")
process.l1tTrackSelectionProducer.debug = options.debug
process.l1tTrackJetsEmulation.VertexInputTag = cms.InputTag("l1tVertexProducer", "l1verticesEmulation")
process.l1tTrackerEmuEtMiss.L1VertexInputTag = cms.InputTag("l1tVertexProducer", "l1verticesEmulation")
process.l1tTrackerEmuEtMiss.debug = options.debug

if options.debug:
    process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(1000000000)
    process.MessageLogger.suppressInfo = cms.untracked.vstring('CondDBESSource', 'PoolDBESSource')
    process.MessageLogger.cerr.CondDBESSource = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )

process.GTTFileWriter.format = cms.untracked.string(options.format)
# process.GTTFileWriter.outputFilename = cms.untracked.string("myOutputFile.txt")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

process.p = cms.Path(process.l1tGTTInputProducer * process.l1tVertexProducer * process.l1tTrackSelectionProducer * process.l1tTrackJetsEmulation * process.l1tTrackerEmuHTMiss * process.l1tTrackerEmuEtMiss * process.GTTFileWriter)
