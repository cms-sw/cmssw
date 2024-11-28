import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.eventIds = cms.EDProducer("edmtest::EventIDProducer")

process.eventValidator = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('eventIds')
)

process.path = cms.Path(process.eventIds + process.eventValidator)
