
import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(3)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    saveFileName = cms.untracked.string('StashStateStream.data'),

    t1 = cms.PSet(
        initialSeed = cms.untracked.uint32(81)
    ),
    t2 = cms.PSet(
        engineName = cms.untracked.string('RanecuEngine'),
        initialSeedSet = cms.untracked.vuint32(1, 2)
    ),
    t3 = cms.PSet(
        initialSeed = cms.untracked.uint32(83),
        engineName = cms.untracked.string('TRandom3')
    ),
    t4 = cms.PSet(
        engineName = cms.untracked.string('HepJamesRandom'),
        initialSeed = cms.untracked.uint32(84)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(191),
        engineName = cms.untracked.string('TRandom3')
    ),
    enableChecking = cms.untracked.bool(True),
    verbose = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3)
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(81),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3)
)
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('RanecuEngine'),
                            seeds = cms.untracked.vuint32(1, 2),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3)
)
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('TRandom3'),
                            seeds = cms.untracked.vuint32(83),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3)
)
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(84),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3)
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testMultiStream.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer)
process.o = cms.EndPath(process.out)
