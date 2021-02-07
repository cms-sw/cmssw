import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(75)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    Thing = cms.PSet(
        initialSeed = cms.untracked.uint32(12345)
    )
)

process.source = cms.Source("PoolSource",
    skipBadFiles = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
        'file:SecondaryInputTest.root',
        'file:SecondaryInputTest.root',
        'file:SecondaryInputTest.root'
    )
)

process.Thing = cms.EDProducer("SecondaryProducer",
    input = cms.SecSource("EmbeddedRootSource",
        skipBadFiles = cms.untracked.bool(True),
        maxFileSkips = cms.untracked.uint32(100),
        fileNames = cms.untracked.vstring(
            'file:SecondaryInputTest2.root',
            'file:missing.root',
            'file:SecondaryInputTest2.root'
        )
    )
)

process.Analysis = cms.EDAnalyzer("EventContentAnalyzer",
    verbose = cms.untracked.bool(False)
)

process.p = cms.Path(process.Thing*process.Analysis)


