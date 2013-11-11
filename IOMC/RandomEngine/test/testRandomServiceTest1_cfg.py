import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRandomServiceMerge1.root'
    ),
    firstRun = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRandomServiceTest1.root')
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
    enableChecking = cms.untracked.bool(True),
    restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(81),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(15),
                            nStreams = cms.untracked.uint32(1),
                            skippedEvents = cms.untracked.vuint32(2),
                            seedByLumi = cms.untracked.vuint32(0, 82, 82, 202, 202)
)
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('RanecuEngine'),
                            seeds = cms.untracked.vuint32(1, 2),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(15),
                            nStreams = cms.untracked.uint32(1),
                            skippedEvents = cms.untracked.vuint32(2),
                            seedByLumi = cms.untracked.vuint32(0, 2, 2, 203, 203)
)
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('TRandom3'),
                            seeds = cms.untracked.vuint32(83),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(15),
                            nStreams = cms.untracked.uint32(1),
                            skippedEvents = cms.untracked.vuint32(2),
                            seedByLumi = cms.untracked.vuint32(0, 84, 84, 204, 204)
)
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(84),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(15),
                            nStreams = cms.untracked.uint32(1),
                            skippedEvents = cms.untracked.vuint32(2),
                            seedByLumi = cms.untracked.vuint32(0, 85, 85, 205, 205)
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4)
process.o = cms.EndPath(process.out)
