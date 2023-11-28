import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing
import pkgutil
import sys


# PART 1 : PARSE ARGUMENTS

options = VarParsing.VarParsing ('analysis')
options.register('redir', 'root://cms-xrd-global.cern.ch/', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "The XRootD redirector to use")
options.register('nstart', 0,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "File index to start on")
options.register('nfiles', -1,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of files to process per job")
options.register('storeTracks', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Store tracks in NTuple")
options.register('l1Tracks','l1tTTTracksFromTrackletEmulation:Level1TTTracks', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, 'L1 track collection to use')
options.register('runVariations', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Run some pre-defined algorithmic variations")
options.register('threads', 1,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of threads to run")
options.register('streams', 0,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of streams to run")
options.register('memoryProfiler', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Run the memory profile")
options.register('tmi', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Run a simple profiler")
options.register('trace', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Dump the paths and consumes")
options.register('dump', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Dump the configuration and exit")
options.parseArguments()

# handle site name usage
if options.redir[0]=="T":
    options.redir = "root://cms-xrd-global.cern.ch//store/test/xrootd/"+options.redir

# Load input files
inputFiles = cms.untracked.vstring()

for filePath in options.inputFiles:
    if filePath.endswith(".root") :    
        inputFiles.append( filePath )
    elif filePath.endswith(".txt"):
        inputFiles += FileUtils.loadListFromFile( filePath )
    elif filePath.endswith("_cff.py"):
        inputFilesImport = getattr(__import__(filePath.strip(".py").strip("python/").replace('/','.'),fromlist=["readFiles"]),"readFiles")
        if options.nfiles==-1:
            inputFiles.extend( inputFilesImport )
        else:
            inputFiles.extend( inputFilesImport[options.nstart:(options.nstart+options.nfiles)] )
    elif pkgutil.find_loader("L1Trigger.VertexFinder."+filePath+"_cff") is not None:
        inputFilesImport = getattr(__import__("L1Trigger.VertexFinder."+filePath+"_cff",fromlist=["readFiles"]),"readFiles")
        if options.nfiles==-1:
            inputFiles.extend( inputFilesImport )
        else:
            inputFiles.extend( inputFilesImport[options.nstart:(options.nstart+options.nfiles)] )
    else:
        raise RuntimeError("Must specify a list of ROOT files, a list of txt files containing a list of ROOT files, or a list of python input files.")

if options.redir != "":
    inputFiles = [(options.redir if val.startswith("/") else "")+val for val in inputFiles]

if options.l1Tracks.count(':') != 1:
    raise RuntimeError("Value for 'l1Tracks' command-line argument (= '{}') should contain one colon".format(options.l1Tracks))

l1TracksTag = cms.InputTag(options.l1Tracks.split(':')[0], options.l1Tracks.split(':')[1])
print "Input Track Collection = {0}  {1}".format(*options.l1Tracks.split(':')) 


# PART 2: SETUP MAIN CMSSW PROCESS 

process = cms.Process("L1TVertexFinder")

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(inputFiles) )
process.TFileService = cms.Service("TFileService", fileName = cms.string(options.outputFile))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.threads),
    numberOfStreams = cms.untracked.uint32(options.streams if options.streams>0 else 0)
)

process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
process.l1tVertexProducer.l1TracksInputTag = l1TracksTag

process.load('L1Trigger.VertexFinder.l1tTPStubValueMapProducer_cfi')
process.load('L1Trigger.VertexFinder.l1tInputDataProducer_cfi')

process.load('L1Trigger.VertexFinder.l1tVertexNTupler_cfi')
process.l1tVertexNTupler.l1TracksInputTag = l1TracksTag

if process.l1tVertexNTupler.debug == 0:
    process.MessageLogger.cerr.FwkReport.reportEvery = 50
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

producerSum = process.l1tVertexProducer
additionalProducerAlgorithms = ["fastHistoEmulation", "fastHistoLooseAssociation", "DBSCAN"]
for algo in additionalProducerAlgorithms:
    producerName = 'VertexProducer{0}'.format(algo)
    producerName = producerName.replace(".","p") # legalize the name

    producer = process.l1tVertexProducer.clone()
    producer.VertexReconstruction.Algorithm = cms.string(algo)

    if "Emulation" in algo:
        if "L1GTTInputProducer" not in process.producerNames():
            process.load('L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi')
            producer.l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted")
            producerSum = process.L1GTTInputProducer + producerSum

        process.l1tVertexNTupler.emulationVertexInputTags.append( cms.InputTag(producerName, 'L1VerticesEmulation') )
        process.l1tVertexNTupler.emulationVertexBranchNames.append(algo)
    else:
        process.l1tVertexNTupler.l1VertexInputTags.append( cms.InputTag(producerName, 'L1Vertices') )
        process.l1tVertexNTupler.l1VertexBranchNames.append(algo)
        process.l1tVertexNTupler.l1VertexTrackInputs.append('hybrid')

    setattr(process, producerName, producer)
    producerSum += producer

# PART 3: PERFORM SCAN OVER ALGO PARAMETER SPACE
if options.runVariations:
    for i in range(1, 9):
        dist = float(i) * 0.05
        for j in range(6):
            minPt = 2.0 + float(j) * 0.2
            for k in range(1, 6):
                minDensity = k
                for l in range(7):
                    seedTrackPt = 2.0 + float(l) * 0.5

                    print
                    print "dist       =", dist
                    print "minPt      =", minPt
                    print "minDensity =", minDensity
                    print "seedTrkPt  =", seedTrackPt

                    producer = process.l1tVertexProducer.clone()
                    producer.VertexReconstruction.VertexDistance = cms.double(dist)
                    producer.VertexReconstruction.VxMinTrackPt = cms.double(minPt)
                    producer.VertexReconstruction.DBSCANMinDensityTracks = cms.uint32(minDensity)
                    producer.VertexReconstruction.DBSCANPtThreshold = cms.double(seedTrackPt)

                    producerName = 'VertexProducerDBSCANDist{0}minPt{1}minDensity{2}seedTrackPt{3}'.format(dist, minPt, minDensity, seedTrackPt)
                    producerName = producerName.replace(".","p")
                    print "producer name =", producerName
                    setattr(process, producerName, producer)
                    producerNames += [producerName]
                    process.l1tVertexNTupler.extraVertexDescriptions += ['DBSCAN(dist={0},minPt={1},minDensity={2},seedTrackPt{3})'.format(dist, minPt, minDensity, seedTrackPt)]
                    process.l1tVertexNTupler.extraVertexInputTags.append( cms.InputTag(producerName, 'L1Vertices'))
                    producerSum += producer

print "Total number of producers =", len(additionalProducerAlgorithms)+1
print "  Producers = [{0}]".format(producerSum.dumpSequenceConfig().replace('&',', '))
print "  Algorithms = [fastHisto, {0}]".format(', '.join(additionalProducerAlgorithms))

# PART 4: UTILITIES

# MEMORY PROFILING
if options.memoryProfiler:
    process.IgProfService = cms.Service("IgProfService",
        reportEventInterval = cms.untracked.int32(1),
        reportFirstEvent = cms.untracked.int32(1),
        reportToFileAtPostEndJob = cms.untracked.string('| gzip -c > '+options.outputFile+'___memory___%I_EndOfJob.gz'),
        reportToFileAtPostEvent = cms.untracked.string('| gzip -c > '+options.outputFile+'___memory___%I.gz')
    )

# SIMPLER PROFILING
if options.tmi:
    from Validation.Performance.TimeMemoryInfo import customise
    process = customise(process)

if options.trace:
    process.add_(cms.Service("Tracer", dumpPathsAndConsumes = cms.untracked.bool(True)))

# SETUP THE PATH
process.p = cms.Path(producerSum + process.l1tTPStubValueMapProducer + process.l1tInputDataProducer + process.l1tVertexNTupler)

# DUMP AND EXIT
if options.dump:
    print process.dumpPython()
    sys.exit(0)
