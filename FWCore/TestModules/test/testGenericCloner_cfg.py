import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.eventIds = cms.EDProducer("edmtest::EventIDProducer")

process.cloneByLabel = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("eventIds"),
    verbose = cms.untracked.bool(True)
)

process.cloneByBranch = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("*_eventIds__TEST"),
    verbose = cms.untracked.bool(True)
)

process.validateByLabel = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('cloneByLabel')
)

process.validateByBranch = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('cloneByBranch')
)

process.task = cms.Task(process.eventIds, process.cloneByLabel, process.cloneByBranch)

process.path = cms.Path(process.validateByLabel + process.validateByBranch, process.task)
