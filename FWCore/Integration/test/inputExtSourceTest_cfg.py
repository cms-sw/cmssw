import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("ThingExtSource",
    module_label = cms.untracked.string('Thing'),
    fileNames = cms.untracked.vstring('file:dummy')
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag('source')
)

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.p = cms.Path(process.OtherThing*process.Analysis)
