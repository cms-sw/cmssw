import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(42)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:SecondaryInputTest.root')
)

process.Thing = cms.EDProducer("SecondaryProducer",
    sequential = cms.untracked.bool(True),
    lumiSpecified = cms.untracked.bool(True),
    input = cms.SecSource("PoolSource",
        fileNames = cms.untracked.vstring('file:SecondaryInputTest2.root')
    )
)

process.Analysis = cms.EDAnalyzer("EventContentAnalyzer",
    verbose = cms.untracked.bool(False)
)

process.p = cms.Path(process.Thing*process.Analysis)


