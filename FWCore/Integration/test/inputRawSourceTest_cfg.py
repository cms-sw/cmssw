import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("ThingRawSource",
    module_label = cms.untracked.string('Thing')
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingLabel = cms.untracked.string('source'),
    debugLevel = cms.untracked.int32(1)
)

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer",
    debugLevel = cms.untracked.int32(1)
)

process.p = cms.Path(process.OtherThing*process.Analysis)
