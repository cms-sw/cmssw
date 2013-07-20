import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    t1 = cms.PSet(
        initialSeed = cms.untracked.uint32(81)
    ),
    t3 = cms.PSet(
        initialSeed = cms.untracked.uint32(83)
    ),
    t4 = cms.PSet(
        initialSeed = cms.untracked.uint32(84)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(100)
    ),
    restoreStateLabel = cms.untracked.string(''),
    saveFileName = cms.untracked.string('test'),
    restoreFileName = cms.untracked.string('')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer",
                            dump = cms.untracked.bool(True))

process.p = cms.Path(process.t1)
