import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource", numberEventsInLuminosityBlock = cms.untracked.uint32(2))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32( 20 ) )

process.options = cms.untracked.PSet( numberOfThreads = cms.untracked.uint32(4),
                                      numberOfStreams = cms.untracked.uint32(0),
                                      numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(2))

process.prod = cms.EDProducer("BusyWaitIntProducer",
                              ivalue = cms.int32(1),
                              iterations = cms.uint32(50*1000*20) )

process.LumiSumIntProd = cms.EDProducer("edmtest::global::LumiSummaryIntProducer",
    transitions = cms.int32(60)
    ,cachevalue = cms.int32(2)
)

process.LumiSumLumiProd = cms.EDProducer("edmtest::global::LumiSummaryLumiProducer",
    transitions = cms.int32(70)
    ,cachevalue = cms.int32(2)
)

process.LumiSumIntFilter = cms.EDFilter("edmtest::global::LumiSummaryIntFilter",
    transitions = cms.int32(84)
    ,cachevalue = cms.int32(2)
)

process.LumiSumIntAnalyzer = cms.EDAnalyzer("edmtest::global::LumiSummaryIntAnalyzer",
    transitions = cms.int32(84)
    ,cachevalue = cms.int32(2)
)

process.LimitedLumiSumIntProd = cms.EDProducer("edmtest::limited::LumiSummaryIntProducer",
    transitions = cms.int32(60)
    ,cachevalue = cms.int32(2)
    ,concurrencyLimit = cms.untracked.uint32(1)
)

process.LimitedLumiSumLumiProd = cms.EDProducer("edmtest::limited::LumiSummaryLumiProducer",
    transitions = cms.int32(70)
    ,cachevalue = cms.int32(2)
    ,concurrencyLimit = cms.untracked.uint32(1)
)

process.LimitedLumiSumIntFilter = cms.EDFilter("edmtest::limited::LumiSummaryIntFilter",
    transitions = cms.int32(84)
    ,cachevalue = cms.int32(2)
    ,concurrencyLimit = cms.untracked.uint32(1)
)

process.LimitedLumiSumIntAnalyzer = cms.EDAnalyzer("edmtest::limited::LumiSummaryIntAnalyzer",
    transitions = cms.int32(84)
    ,cachevalue = cms.int32(2)
    ,concurrencyLimit = cms.untracked.uint32(1)
)

process.p = cms.Path(process.prod * process.LumiSumIntProd * process.LumiSumLumiProd * process.LumiSumIntFilter * process.LumiSumIntAnalyzer * process.LimitedLumiSumIntProd * process.LimitedLumiSumLumiProd * process.LimitedLumiSumIntFilter * process.LimitedLumiSumIntAnalyzer)

process.add_(cms.Service("ConcurrentModuleTimer",
                         modulesToExclude = cms.untracked.vstring("TriggerResults","p"),
                         excludeSource = cms.untracked.bool(True)))

