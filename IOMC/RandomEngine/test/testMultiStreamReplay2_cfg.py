
import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    t1 = cms.PSet(
        initialSeed = cms.untracked.uint32(7)
    ),
    t2 = cms.PSet(
        engineName = cms.untracked.string('RanecuEngine'),
        initialSeedSet = cms.untracked.vuint32(7, 7)
    ),
    t3 = cms.PSet(
        initialSeed = cms.untracked.uint32(7),
        engineName = cms.untracked.string('TRandom3')
    ),
    t4 = cms.PSet(
        engineName = cms.untracked.string('HepJamesRandom'),
        initialSeed = cms.untracked.uint32(7)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(7),
        engineName = cms.untracked.string('TRandom3')
    ),
    restoreFileName = cms.untracked.string('StashStateStream.data_2'),
    enableChecking = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
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
                            nStreams = cms.untracked.uint32(1),
                            seedByLumi = cms.untracked.vuint32(0, 84, 84, 204, 204),
                            multiStreamReplay = cms.untracked.bool(True)
)
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('RanecuEngine'),
                            seeds = cms.untracked.vuint32(1, 2),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1),
                            seedByLumi = cms.untracked.vuint32(0, 4, 4, 205, 205),
                            multiStreamReplay = cms.untracked.bool(True)
)
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('TRandom3'),
                            seeds = cms.untracked.vuint32(83),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1),
                            seedByLumi = cms.untracked.vuint32(0, 86, 86, 206, 206),
                            multiStreamReplay = cms.untracked.bool(True)
)
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(84),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1),
                            seedByLumi = cms.untracked.vuint32(0, 87, 87, 207, 207),
                            multiStreamReplay = cms.untracked.bool(True)
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testMultiStreamReplay2.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer)
process.o = cms.EndPath(process.out)
