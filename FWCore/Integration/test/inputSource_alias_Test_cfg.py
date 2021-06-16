import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("ThingSource"
)

process.thing = cms.EDAlias(source = cms.VPSet( cms.PSet( type = cms.string("edmtestThings")) ) )


process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag('thing')
)

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.p = cms.Path(process.OtherThing * process.Analysis)
