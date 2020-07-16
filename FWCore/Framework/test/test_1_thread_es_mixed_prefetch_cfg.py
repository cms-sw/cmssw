import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.b = cms.EDProducer("edmtest::one::WatchRunsProducer",
                           transitions = cms.int32(3))
process.c = cms.EDAnalyzer("TestESDummyDataAnalyzer",
                           expected = cms.int32(5))
process.a = cms.EDAnalyzer("TestESDummyDataNoPrefetchAnalyzer")

process.add_(cms.ESProducer("LoadableDummyProvider",
                            value = cms.untracked.int32(5)))

process.essource = cms.ESSource("EmptyESSource",
    recordName = cms.string('DummyRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.p1 = cms.Path(process.a)
process.p2 = cms.Path(process.b)
process.p3 = cms.Path(process.c)

#process.add_(cms.Service("Tracer"))

process.add_(cms.Service("ConcurrentModuleTimer", padding = cms.untracked.uint32(2), trackGlobalBeginRun = cms.untracked.bool(True)))
