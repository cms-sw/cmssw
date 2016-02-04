import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

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

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4)
process.o = cms.EndPath(process.out)
