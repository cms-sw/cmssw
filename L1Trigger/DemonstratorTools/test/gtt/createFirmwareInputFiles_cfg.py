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
                  "File format (APx, EMP or X20)")
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

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(inputFiles) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.threads),
    numberOfStreams = cms.untracked.uint32(options.streams if options.streams>0 else 0)
)

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
process.load('L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi')
process.load('L1Trigger.VertexFinder.VertexProducer_cff')
process.load('L1Trigger.DemonstratorTools.GTTFileWriter_cff')

process.L1GTTInputProducer.debug = cms.int32(options.debug)
process.VertexProducer.l1TracksInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted")
process.VertexProducer.VertexReconstruction.Algorithm = cms.string("FastHistoEmulation")
process.VertexProducer.VertexReconstruction.VxMinTrackPt = cms.double(0.0)
process.VertexProducer.debug = options.debug
if options.debug:
    process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(1000000000)

process.GTTFileWriter.format = cms.untracked.string(options.format)
# process.GTTFileWriter.outputFilename = cms.untracked.string("myOutputFile.txt")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

process.p = cms.Path(process.L1HybridTracks * process.L1GTTInputProducer * process.VertexProducer * process.GTTFileWriter)
