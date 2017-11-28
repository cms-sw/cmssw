import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("ThingSource",
    module_label = cms.untracked.string('Thing')
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag('source')
)

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.p = cms.Path(process.OtherThing * process.Analysis)
