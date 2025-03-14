import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.fail = cms.EDProducer("edmtest::FailingInRunProducer")

process.tstStream = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer",
                                   transitions=cms.int32(2),
                                   nLumis = cms.untracked.uint32(0))
process.tstGlobal = cms.EDAnalyzer("edmtest::global::RunIntAnalyzer",
                                   transitions=cms.int32(2),
                                   cachevalue = cms.int32(0))

process.p = cms.Path(process.fail+process.tstStream+process.tstGlobal)

process.add_(cms.Service("Tracer"))
