import FWCore.ParameterSet.Config as cms

process = cms.Process("A")

process.maxEvents.input = 1
process.source = cms.Source("EmptySource")

process.load("FWCore.Services.DependencyGraph_cfi")
process.DependencyGraph.fileName = "test_module_delete_dependencygraph.gv"

# Each of these modules declares it produces a product that is not consumed.
# The TestModuleDeleteAnalyzer will test in its beginJob transition that the
# producer module was deleted.
process.producerEventNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerBeginLumiNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer")
process.producerBeginRunNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer")
process.producerBeginProcessNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInProcessProducer")

process.intAnalyzerDelete = cms.EDAnalyzer("edmtest::TestModuleDeleteAnalyzer")

process.t = cms.Task(
    process.producerEventNotConsumed,
    process.producerBeginLumiNotConsumed,
    process.producerBeginRunNotConsumed,
    process.producerBeginProcessNotConsumed
)

process.p = cms.Path(process.intAnalyzerDelete, process.t)
