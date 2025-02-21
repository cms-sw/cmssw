import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

# produce and validate products of type int
process.produceInt = cms.EDProducer("edmtest::GlobalIntProducer",
    value = cms.int32(42)
)

process.validateInt = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("produceInt"),
    valueMustMatch = cms.untracked.int32(42)
)

process.taskInt = cms.Task(process.produceInt)

process.pathInt = cms.Path(process.validateInt, process.taskInt)

# produce and validate products of type std::string
process.produceString = cms.EDProducer("edmtest::GlobalStringProducer",
    value = cms.string("Hello world")
)

process.validateString = cms.EDAnalyzer("edmtest::GlobalStringAnalyzer",
    source = cms.InputTag("produceString"),
    expected = cms.string("Hello world")
)

process.taskString = cms.Task(process.produceString)

process.pathString = cms.Path(process.validateString, process.taskString)
