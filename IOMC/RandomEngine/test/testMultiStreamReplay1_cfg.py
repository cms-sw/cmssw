
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(3)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testMultiStream.root'
    ),
    firstRun = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(3)
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
    verbose = cms.untracked.bool(False),
    restoreStateTag = cms.untracked.InputTag('randomEngineStateProducer')
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(81),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3),
                            multiStreamReplay = cms.untracked.bool(True)
)
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('RanecuEngine'),
                            seeds = cms.untracked.vuint32(1, 2),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3),
                            multiStreamReplay = cms.untracked.bool(True)
)
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('TRandom3'),
                            seeds = cms.untracked.vuint32(83),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3),
                            multiStreamReplay = cms.untracked.bool(True)
)
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceGlobal",
                            engineName = cms.untracked.string('HepJamesRandom'),
                            seeds = cms.untracked.vuint32(84),
                            offset = cms.untracked.uint32(0),
                            maxEvents = cms.untracked.uint32(5),
                            nStreams = cms.untracked.uint32(3),
                            multiStreamReplay = cms.untracked.bool(True)
)

process.randomEngineStateProducer2 = cms.EDProducer("RandomEngineStateProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testMultiStreamReplay1.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer2)
process.o = cms.EndPath(process.out)
