import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

# The following two lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.p = cms.Path(process.Thing * process.OtherThing * process.Analysis)


