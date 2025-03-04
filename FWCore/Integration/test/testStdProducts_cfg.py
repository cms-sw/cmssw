import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

# produce and validate products of type int
intValue = 42

process.produceInt = cms.EDProducer("edmtest::GlobalIntProducer",
    value = cms.int32(intValue)
)

process.validateInt = cms.EDAnalyzer("edmtest::GlobalIntAnalyzer",
    source = cms.InputTag("produceInt"),
    expected = cms.int32(intValue)
)

process.taskInt = cms.Task(process.produceInt)

process.pathInt = cms.Path(process.validateInt, process.taskInt)

# produce and validate products of type float
floatValue = 3.14159

process.produceFloat = cms.EDProducer("edmtest::GlobalFloatProducer",
    value = cms.double(floatValue)
)

process.validateFloat = cms.EDAnalyzer("edmtest::GlobalFloatAnalyzer",
    source = cms.InputTag("produceFloat"),
    expected = cms.double(floatValue)
)

process.taskFloat = cms.Task(process.produceFloat)

process.pathFloat = cms.Path(process.validateFloat, process.taskFloat)

# produce and validate products of type std::string
stringValue = "Hello world"

process.produceString = cms.EDProducer("edmtest::GlobalStringProducer",
    value = cms.string(stringValue)
)

process.validateString = cms.EDAnalyzer("edmtest::GlobalStringAnalyzer",
    source = cms.InputTag("produceString"),
    expected = cms.string(stringValue)
)

process.taskString = cms.Task(process.produceString)

process.pathString = cms.Path(process.validateString, process.taskString)

# produce and validate products of type std::vector<double>
import random
random.seed()
doubleValues = [ random.uniform(-10., 10.) for i in range(10) ]

process.produceVector = cms.EDProducer("edmtest::GlobalVectorProducer",
    values = cms.vdouble(doubleValues)
)

process.validateVector = cms.EDAnalyzer("edmtest::GlobalVectorAnalyzer",
    source = cms.InputTag("produceVector"),
    expected = cms.vdouble(doubleValues)
)

process.taskVector = cms.Task(process.produceVector)

process.pathVector = cms.Path(process.validateVector, process.taskVector)
