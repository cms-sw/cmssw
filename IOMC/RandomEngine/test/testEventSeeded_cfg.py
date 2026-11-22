import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register("streams", 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "number of streams and threads")
options.register("firstRun", 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "run number")
options.register("firstLumi", 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "first luminosity block")
options.register("events", 12, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "number of events")
options.register("seedOffset", 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 "value added to every initialSeed")
options.register("seedingMode", "event", VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string, "seedingMode of the RandomNumberGeneratorService")
options.register("input", "", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 "replay the engine states of this file")
options.register("output", "", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 "write the engine states to this file")
options.register("philoxEngine", "PhiloxEngine", VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string, "engine of the module philox")
options.register("addedModule", False, VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool, "add a module with the seed of mixMax and no stored state")
options.parseArguments()

process = cms.Process("EVENTSEEDED" if not options.input else "REPLAY")

process.options.numberOfThreads = options.streams
process.options.numberOfStreams = options.streams
# Two lumis at the same time, so that two lumi engines are in use together.
process.options.numberOfConcurrentLuminosityBlocks = 2

# 12 events, 3 per lumi: 4 lumis. The event numbers continue over the lumis, so a job
# that starts at a later lumi gives the same EventIDs as the full job.
eventsPerLumi = 3
if options.input:
    process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:" + options.input))
else:
    process.source = cms.Source("EmptySource",
                                firstRun=cms.untracked.uint32(options.firstRun),
                                firstLuminosityBlock=cms.untracked.uint32(options.firstLumi),
                                firstEvent=cms.untracked.uint32((options.firstLumi - 1) * eventsPerLumi + 1),
                                numberEventsInLuminosityBlock=cms.untracked.uint32(eventsPerLumi))
    process.maxEvents.input = options.events

# mixMax and philox have the same seed, so only the module label separates their numbers.
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    seedingMode=cms.untracked.string(options.seedingMode),
    enableChecking=cms.untracked.bool(True),
    mixMax=cms.PSet(initialSeed=cms.untracked.uint32(1234567 + options.seedOffset),
                    engineName=cms.untracked.string("MixMaxRng")),
    philox=cms.PSet(initialSeed=cms.untracked.uint32(1234567 + options.seedOffset),
                    engineName=cms.untracked.string(options.philoxEngine)),
    randomEngineStateProducer=cms.PSet(initialSeed=cms.untracked.uint32(1),
                                       engineName=cms.untracked.string("MixMaxRng")))
if options.input:
    process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string("randomEngineStateProducer")

process.mixMax = cms.EDAnalyzer("TestEventSeededRandom")
process.philox = cms.EDAnalyzer("TestEventSeededRandom")
process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.p = cms.Path(process.randomEngineStateProducer + process.mixMax + process.philox)

if options.addedModule:
    process.RandomNumberGeneratorService.added = process.RandomNumberGeneratorService.mixMax.clone()
    process.added = cms.EDAnalyzer("TestEventSeededRandom")
    process.p += process.added

if options.output:
    process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string(options.output))
    process.e = cms.EndPath(process.out)
