import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    saveFileName = cms.untracked.string('StashState2.data'),

    t1 = cms.PSet(
        initialSeed = cms.untracked.uint32(201)
    ),
    t2 = cms.PSet(
        engineName = cms.untracked.string('RanecuEngine'),
        initialSeedSet = cms.untracked.vuint32(202, 2)
    ),
    t3 = cms.PSet(
        initialSeed = cms.untracked.uint32(203),
        engineName = cms.untracked.string('TRandom3')
    ),
    t4 = cms.PSet(
        engineName = cms.untracked.string('HepJamesRandom'),
        initialSeed = cms.untracked.uint32(204)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(205),
        engineName = cms.untracked.string('TRandom3')
    ),
    enableChecking = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(3),
    firstEvent = cms.untracked.uint32(6),
    numberEventsInRun = cms.untracked.uint32(100),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3)
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(201),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1)
)
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('RanecuEngine'),
                            seeds = cms.untracked.vuint32(202, 2),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1)
)
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('TRandom3'),
                            seeds = cms.untracked.vuint32(203),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1)
)
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(204),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(1)
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRandomService2.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer)
process.o = cms.EndPath(process.out)
