import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTANALYSIS")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
    setRunNumber = cms.untracked.uint32(621),
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag("AltThing")
)

process.p = cms.Path(process.OtherThing*process.Analysis)


