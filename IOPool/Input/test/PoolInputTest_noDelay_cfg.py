import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.OtherThing = cms.EDProducer("OtherThingProducer")

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
                            delayReadingEventProducts = cms.untracked.bool(False),
    setRunNumber = cms.untracked.uint32(621),
    fileNames = cms.untracked.vstring('file:PoolInputTest.root')
)

process.p = cms.Path(process.OtherThing*process.Analysis)

process.add_(cms.Service("Tracer"))
