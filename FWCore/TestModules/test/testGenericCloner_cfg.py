import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

# produce, clone and validate products of type int
process.produceInt = cms.EDProducer("edmtest::GlobalIntProducer",
    value = cms.int32(42)
)

process.cloneInt = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("produceInt"),
    verbose = cms.untracked.bool(True)
)

process.validateInt = cms.EDAnalyzer("edmtest::GlobalIntAnalyzer",
    source = cms.InputTag("cloneInt"),
    expected = cms.int32(42)
)

process.taskInt = cms.Task(process.produceInt, process.cloneInt)

process.pathInt = cms.Path(process.validateInt, process.taskInt)

# produce, clone and validate products of type std::string
process.produceString = cms.EDProducer("edmtest::GlobalStringProducer",
    value = cms.string("Hello world")
)

process.cloneString = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("produceString"),
    verbose = cms.untracked.bool(True)
)

process.validateString = cms.EDAnalyzer("edmtest::GlobalStringAnalyzer",
    source = cms.InputTag("cloneString"),
    expected = cms.string("Hello world")
)

process.taskString = cms.Task(process.produceString, process.cloneString)

process.pathString = cms.Path(process.validateString, process.taskString)

# produce, clone and validate products of type edm::EventID
process.eventIds = cms.EDProducer("edmtest::EventIDProducer")

process.cloneIdsByLabel = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("eventIds"),
    verbose = cms.untracked.bool(True)
)

process.cloneIdsByBranch = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("*_eventIds__TEST"),
    verbose = cms.untracked.bool(True)
)

process.validateIdsByLabel = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('cloneIdsByLabel')
)

process.validateIdsByBranch = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('cloneIdsByBranch')
)

process.taskIds = cms.Task(process.eventIds, process.cloneIdsByLabel, process.cloneIdsByBranch)

process.pathIds = cms.Path(process.validateIdsByLabel + process.validateIdsByBranch, process.taskIds)

# will not clone a transient product
process.produceTransient = cms.EDProducer("TransientIntProducer",
    ivalue = cms.int32(22)
)

process.cloneTransient = cms.EDProducer("edmtest::GenericCloner",
    eventProducts = cms.vstring("produceTransient"),
    verbose = cms.untracked.bool(True)
)

process.pathTransient = cms.Path(process.produceTransient + process.cloneTransient)
