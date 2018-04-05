import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.fail = cms.EDProducer("edmtest::FailingInRunProducer")

process.tstStream = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions=cms.int32(2))
process.tstGlobal = cms.EDAnalyzer("edmtest::global::RunIntAnalyzer",
                                   transitions=cms.int32(2),
                                   cachevalue = cms.int32(0))

process.p = cms.Path(process.fail+process.tstStream+process.tstGlobal)

process.add_(cms.Service("Tracer"))

process2 = cms.Process("Test2")
process2.tstStreamSub = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions=cms.int32(2))
process2.tstGlobalSub = cms.EDAnalyzer("edmtest::global::RunIntAnalyzer",
                                   transitions=cms.int32(2),
                                   cachevalue = cms.int32(0))
process2.p2 = cms.Path(process2.tstStreamSub+process2.tstGlobalSub)
process.addSubProcess(cms.SubProcess(process2))
