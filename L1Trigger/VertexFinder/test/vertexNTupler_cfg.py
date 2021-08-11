import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing


# PART 1 : PARSE ARGUMENTS

options = VarParsing.VarParsing ('analysis')
options.register('storeTracks', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Store tracks in NTuple")
options.register('l1Tracks','TTTracksFromTrackletEmulation:Level1TTTracks', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, 'L1 track collection to use')
options.register('runVariations', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Run some pre-defined algorithmic variations")
options.register('threads',1,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of threads/streams to run")
options.parseArguments()

inputFiles = []
for filePath in options.inputFiles:
    if filePath.endswith(".root"):
        inputFiles.append(filePath)
    else:
        inputFiles += FileUtils.loadListFromFile(filePath)

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
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(inputFiles) )
process.TFileService = cms.Service("TFileService", fileName = cms.string(options.outputFile))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.threads),
    numberOfStreams = cms.untracked.uint32(options.threads if options.threads>0 else 0)
)

process.load('L1Trigger.VertexFinder.VertexProducer_cff')
process.VertexProducer.l1TracksInputTag = l1TracksTag

process.load('L1Trigger.VertexFinder.TPStubValueMapProducer_cff')
process.load('L1Trigger.VertexFinder.InputDataProducer_cff')

process.load('L1Trigger.VertexFinder.VertexNTupler_cff')
process.L1TVertexNTupler.l1TracksInputTag = l1TracksTag

if process.L1TVertexNTupler.debug == 0:
    process.MessageLogger.cerr.FwkReport.reportEvery = 50
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

producerSum = process.VertexProducer
additionalProducerAlgorithms = ["FastHistoLooseAssociation", "DBSCAN"]
for algo in additionalProducerAlgorithms:
    producerName = 'VertexProducer{0}'.format(algo)
    producerName = producerName.replace(".","p") # legalize the name

    producer = process.VertexProducer.clone()
    producer.VertexReconstruction.Algorithm = cms.string(algo)
    setattr(process, producerName, producer)
    producerSum += producer

    process.L1TVertexNTupler.l1VertexInputTags.append( cms.InputTag(producerName, 'l1vertices') )
    process.L1TVertexNTupler.l1VertexBranchNames.append(algo)
    process.L1TVertexNTupler.l1VertexTrackInputs.append('hybrid')

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

                    producer = process.VertexProducer.clone()
                    producer.VertexReconstruction.VertexDistance = cms.double(dist)
                    producer.VertexReconstruction.VxMinTrackPt = cms.double(minPt)
                    producer.VertexReconstruction.DBSCANMinDensityTracks = cms.uint32(minDensity)
                    producer.VertexReconstruction.DBSCANPtThreshold = cms.double(seedTrackPt)

                    producerName = 'VertexProducerDBSCANDist{0}minPt{1}minDensity{2}seedTrackPt{3}'.format(dist, minPt, minDensity, seedTrackPt)
                    producerName = producerName.replace(".","p")
                    print "producer name =", producerName
                    setattr(process, producerName, producer)
                    producerNames += [producerName]
                    process.L1TVertexNTupler.extraVertexDescriptions += ['DBSCAN(dist={0},minPt={1},minDensity={2},seedTrackPt{3})'.format(dist, minPt, minDensity, seedTrackPt)]
                    process.L1TVertexNTupler.extraVertexInputTags.append( cms.InputTag(producerName, 'l1vertices'))
                    producerSum += producer

print "Total number of producers =", len(additionalProducerAlgorithms)+1
print "  Producers = [{0}]".format(producerSum.dumpSequenceConfig().replace('&',', '))
print "  Algorithms = [FastHisto, {0}]".format(', '.join(additionalProducerAlgorithms))
 
process.p = cms.Path(producerSum + process.TPStubValueMapProducer + process.InputDataProducer + process.L1TVertexNTupler)

