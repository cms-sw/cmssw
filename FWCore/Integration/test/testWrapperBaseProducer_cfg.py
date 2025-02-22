import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.producer = cms.EDProducer("WrapperBaseProducer")

process.validateInt = cms.EDAnalyzer("edmtest::GlobalIntAnalyzer",
    source = cms.InputTag("producer"),
    expected = cms.int32(42)        # hard-coded in WrapperBaseProducer
)

process.validateFloat = cms.EDAnalyzer("edmtest::GlobalFloatAnalyzer",
    source = cms.InputTag("producer"),
    expected = cms.double(3.14159)  # hard-coded in WrapperBaseProducer
)

process.validateString = cms.EDAnalyzer("edmtest::GlobalStringAnalyzer",
    source = cms.InputTag("producer"),
    expected = cms.string("42")     # hard-coded in WrapperBaseProducer
)

process.validateVector = cms.EDAnalyzer("edmtest::GlobalVectorAnalyzer",
    source = cms.InputTag("producer"),
    expected = cms.vdouble(1., 1., 2., 3., 5., 8., 11., 19., 30.)   # hard-coded in WrapperBaseProducer
)

process.path = cms.Path(
    process.producer +
    process.validateInt +
    process.validateFloat +
    process.validateString +
    process.validateVector
)
