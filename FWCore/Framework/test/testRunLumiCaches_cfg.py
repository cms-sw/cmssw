import FWCore.ParameterSet.Config as cms

# These next 4 variables can be modified and the tests should still pass
nStreams = 4
nRuns = 17
nLumisPerRun = 1
nEventsPerLumi = 6

nEventsPerRun = nLumisPerRun*nEventsPerLumi
nLumis = nRuns*nLumisPerRun
nEvents = nRuns*nEventsPerRun

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(nEventsPerLumi),
    numberEventsInRun = cms.untracked.uint32(nEventsPerRun)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nEvents)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(nStreams),
    numberOfConcurrentRuns = cms.untracked.uint32(4),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(4)
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.globalRunIntProd = cms.EDProducer("edmtest::global::RunIntProducer",
    transitions = cms.int32(2*nRuns)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.globalRunSumIntProd = cms.EDProducer("edmtest::global::RunSummaryIntProducer",
    transitions = cms.int32(nRuns*nStreams+2*nRuns)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.globalLumiIntProd = cms.EDProducer("edmtest::global::LumiIntProducer",
    transitions = cms.int32(2*nLumis)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.globalLumiSumIntProd = cms.EDProducer("edmtest::global::LumiSummaryIntProducer",
    transitions = cms.int32(2*nLumis+nStreams*nLumis)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.globalRunIntFilt = cms.EDFilter("edmtest::global::RunIntFilter",
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.globalRunSumIntFilt = cms.EDFilter("edmtest::global::RunSummaryIntFilter",
    transitions = cms.int32(nStreams+nStreams*nRuns+2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.globalLumiIntFilt = cms.EDFilter("edmtest::global::LumiIntFilter",
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.globalLumiSumIntFilt = cms.EDFilter("edmtest::global::LumiSummaryIntFilter",
    transitions = cms.int32(nStreams+nStreams*nLumis+2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.globalRunIntAna = cms.EDAnalyzer("edmtest::global::RunIntAnalyzer",
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.globalRunSumIntAna = cms.EDAnalyzer("edmtest::global::RunSummaryIntAnalyzer",
    transitions = cms.int32(nStreams+nStreams*nRuns+2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.globalLumiIntAna = cms.EDAnalyzer("edmtest::global::LumiIntAnalyzer",
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
    ,moduleLabel = cms.InputTag("TestAccumulator1")
)

process.globalLumiSumIntAna = cms.EDAnalyzer("edmtest::global::LumiSummaryIntAnalyzer",
    transitions = cms.int32(nStreams+nStreams*nLumis+2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.limitedRunIntProd = cms.EDProducer("edmtest::limited::RunIntProducer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*nRuns)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.limitedRunSumIntProd = cms.EDProducer("edmtest::limited::RunSummaryIntProducer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams*nRuns+2*nRuns)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.limitedLumiIntProd = cms.EDProducer("edmtest::limited::LumiIntProducer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*nLumis)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.limitedLumiSumIntProd = cms.EDProducer("edmtest::limited::LumiSummaryIntProducer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams*nLumis+2*nLumis)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.limitedRunIntFilt = cms.EDFilter("edmtest::limited::RunIntFilter",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.limitedRunSumIntFilt = cms.EDFilter("edmtest::limited::RunSummaryIntFilter",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams+nStreams*nRuns+2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.limitedLumiIntFilt = cms.EDFilter("edmtest::limited::LumiIntFilter",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.limitedLumiSumIntFilt = cms.EDFilter("edmtest::limited::LumiSummaryIntFilter",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams+nStreams*nLumis+2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.limitedRunIntAna = cms.EDAnalyzer("edmtest::limited::RunIntAnalyzer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.limitedRunSumIntAna = cms.EDAnalyzer("edmtest::limited::RunSummaryIntAnalyzer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams+nStreams*nRuns+2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.limitedLumiIntAna = cms.EDAnalyzer("edmtest::limited::LumiIntAnalyzer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
    ,moduleLabel = cms.InputTag("TestAccumulator1")
)

process.limitedLumiSumIntAna = cms.EDAnalyzer("edmtest::limited::LumiSummaryIntAnalyzer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams+nStreams*nLumis+2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.oneRunCacheProd = cms.EDProducer("edmtest::one::RunCacheProducer",
    transitions = cms.int32(2*nRuns+nEvents)
)

process.oneLumiBlockCacheProd = cms.EDProducer("edmtest::one::LumiBlockCacheProducer",
    transitions = cms.int32(2*nLumis+nEvents)
)

process.oneRunCacheFilt = cms.EDFilter("edmtest::one::RunCacheFilter",
    transitions = cms.int32(2*nRuns+nEvents)
)

process.oneLumiBlockCacheFilt = cms.EDFilter("edmtest::one::LumiBlockCacheFilter",
    transitions = cms.int32(2*nRuns+nEvents)
)

process.oneRunCacheAna = cms.EDAnalyzer("edmtest::one::RunCacheAnalyzer",
    transitions = cms.int32(2*nRuns+nEvents)
)

process.oneLumiBlockCacheAna = cms.EDAnalyzer("edmtest::one::LumiBlockCacheAnalyzer",
    transitions = cms.int32(2*nRuns+nEvents)
)

process.streamRunIntProd = cms.EDProducer("edmtest::stream::RunIntProducer",
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.streamLumiIntProd = cms.EDProducer("edmtest::stream::LumiIntProducer",
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.streamRunSumIntProd = cms.EDProducer("edmtest::stream::RunSummaryIntProducer",
    transitions = cms.int32(4*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.streamLumiSumIntProd = cms.EDProducer("edmtest::stream::LumiSummaryIntProducer",
    transitions = cms.int32(4*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.streamRunIntFilt = cms.EDFilter("edmtest::stream::RunIntFilter",
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.streamLumiIntFilt = cms.EDFilter("edmtest::stream::LumiIntFilter",
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.streamRunSumIntFilt = cms.EDFilter("edmtest::stream::RunSummaryIntFilter",
    transitions = cms.int32(4*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.streamLumiSumIntFilt = cms.EDFilter("edmtest::stream::LumiSummaryIntFilter",
    transitions = cms.int32(4*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.streamRunIntAna = cms.EDAnalyzer("edmtest::stream::RunIntAnalyzer",
    transitions = cms.int32(2*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.streamLumiIntAna = cms.EDAnalyzer("edmtest::stream::LumiIntAnalyzer",
    transitions = cms.int32(2*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
    ,moduleLabel = cms.InputTag("TestAccumulator1")
)

process.streamRunSumIntAna = cms.EDAnalyzer("edmtest::stream::RunSummaryIntAnalyzer",
    transitions = cms.int32(4*nRuns+nEvents)
    ,cachevalue = cms.int32(nEventsPerRun)
)

process.streamLumiSumIntAna = cms.EDAnalyzer("edmtest::stream::LumiSummaryIntAnalyzer",
    transitions = cms.int32(4*nLumis+nEvents)
    ,cachevalue = cms.int32(nEventsPerLumi)
)

process.path1 = cms.Path(
    process.busy1 *
    process.globalRunIntProd *
    process.globalRunSumIntProd *
    process.globalLumiIntProd *
    process.globalLumiSumIntProd *
    process.globalRunIntFilt *
    process.globalRunSumIntFilt *
    process.globalLumiIntFilt *
    process.globalLumiSumIntFilt *
    process.globalRunIntAna *
    process.globalRunSumIntAna *
    process.globalLumiIntAna *
    process.globalLumiSumIntAna *
    process.limitedRunIntProd *
    process.limitedRunSumIntProd *
    process.limitedLumiIntProd *
    process.limitedLumiSumIntProd *
    process.limitedRunIntFilt *
    process.limitedRunSumIntFilt *
    process.limitedLumiIntFilt *
    process.limitedLumiSumIntFilt *
    process.limitedRunIntAna *
    process.limitedRunSumIntAna *
    process.limitedLumiIntAna *
    process.limitedLumiSumIntAna *
    process.oneRunCacheProd *
    process.oneLumiBlockCacheProd *
    process.oneRunCacheFilt *
    process.oneLumiBlockCacheFilt *
    process.oneRunCacheAna *
    process.oneLumiBlockCacheAna *
    process.streamRunIntProd *
    process.streamLumiIntProd *
    process.streamRunSumIntProd *
    process.streamLumiSumIntProd *
    process.streamRunIntFilt *
    process.streamLumiIntFilt *
    process.streamRunSumIntFilt *
    process.streamLumiSumIntFilt *
    process.streamRunIntAna *
    process.streamLumiIntAna *
    process.streamRunSumIntAna *
    process.streamLumiSumIntAna
)
