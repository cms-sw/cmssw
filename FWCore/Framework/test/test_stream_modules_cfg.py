import FWCore.ParameterSet.Config as cms

nEvtLumi = 4
nEvtRun = 2*nEvtLumi
nRuns = 64
nStreams = 4
nEvt = nRuns*nEvtRun

process = cms.Process("TESTSTREAMMODULES")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(nStreams),
    numberOfThreads = cms.untracked.uint32(nStreams)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nEvt)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint64(1000),
    firstTime = cms.untracked.uint64(1000000),
    numberEventsInRun = cms.untracked.uint32(nEvtRun),
    numberEventsInLuminosityBlock = cms.untracked.uint32(nEvtLumi)
)

#process.Tracer = cms.Service("Tracer")


process.GlobIntProd = cms.EDProducer("edmtest::stream::GlobalIntProducer",
    transitions = cms.int32(nEvt+3)
    ,cachevalue = cms.int32(nEvt)
)

process.RunIntProd = cms.EDProducer("edmtest::stream::RunIntProducer",
    transitions = cms.int32(int(nEvt+2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntProd = cms.EDProducer("edmtest::stream::LumiIntProducer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntProd = cms.EDProducer("edmtest::stream::RunSummaryIntProducer",
    transitions = cms.int32(nEvt+4*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntProd = cms.EDProducer("edmtest::stream::LumiSummaryIntProducer",
    transitions = cms.int32(nEvt+4*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.ProcessBlockIntProd = cms.EDProducer("edmtest::stream::ProcessBlockIntProducer",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.TestBeginProcessBlockProd = cms.EDProducer("edmtest::stream::TestBeginProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("")
)

process.TestBeginProcessBlockProdRead = cms.EDProducer("edmtest::stream::TestBeginProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin")
)

process.TestEndProcessBlockProd = cms.EDProducer("edmtest::stream::TestEndProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("")
)

process.TestEndProcessBlockProdRead = cms.EDProducer("edmtest::stream::TestEndProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.ProcessBlockIntProdNoGlobalCache = cms.EDProducer("edmtest::stream::ProcessBlockIntProducerNoGlobalCache")
process.TestBeginProcessBlockProdNoGlobalCache = cms.EDProducer("edmtest::stream::TestBeginProcessBlockProducerNoGlobalCache")
process.TestEndProcessBlockProdNoGlobalCache = cms.EDProducer("edmtest::stream::TestEndProcessBlockProducerNoGlobalCache")

process.TestBeginRunProd = cms.EDProducer("edmtest::stream::TestBeginRunProducer",
    transitions = cms.int32(int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndRunProd = cms.EDProducer("edmtest::stream::TestEndRunProducer",
    transitions = cms.int32(int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestBeginLumiBlockProd = cms.EDProducer("edmtest::stream::TestBeginLumiBlockProducer",
    transitions = cms.int32(int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndLumiBlockProd = cms.EDProducer("edmtest::stream::TestEndLumiBlockProducer",
    transitions = cms.int32(int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)


process.GlobIntAn = cms.EDAnalyzer("edmtest::stream::GlobalIntAnalyzer",
    transitions = cms.int32(nEvt+3)
    ,cachevalue = cms.int32(nEvt)
)

process.RunIntAn= cms.EDAnalyzer("edmtest::stream::RunIntAnalyzer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntAn = cms.EDAnalyzer("edmtest::stream::LumiIntAnalyzer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
    # needed to avoid deleting TestAccumulator1
    ,moduleLabel = cms.InputTag("TestAccumulator1")
)

process.RunSumIntAn = cms.EDAnalyzer("edmtest::stream::RunSummaryIntAnalyzer",
    transitions = cms.int32(nEvt+4*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntAn = cms.EDAnalyzer("edmtest::stream::LumiSummaryIntAnalyzer",
    transitions = cms.int32(nEvt+4*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.ProcessBlockIntAn = cms.EDAnalyzer("edmtest::stream::ProcessBlockIntAnalyzer",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.GlobIntFil = cms.EDFilter("edmtest::stream::GlobalIntFilter",
    transitions = cms.int32(nEvt+3)
    ,cachevalue = cms.int32(nEvt)
)

process.RunIntFil = cms.EDFilter("edmtest::stream::RunIntFilter",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntFil = cms.EDFilter("edmtest::stream::LumiIntFilter",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntFil = cms.EDFilter("edmtest::stream::RunSummaryIntFilter",
    transitions = cms.int32(nEvt+4*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntFil = cms.EDFilter("edmtest::stream::LumiSummaryIntFilter",
    transitions = cms.int32(nEvt+4*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.ProcessBlockIntFil = cms.EDFilter("edmtest::stream::ProcessBlockIntFilter",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockFil" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockFil", "end")
)

process.TestBeginProcessBlockFil = cms.EDFilter("edmtest::stream::TestBeginProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("")
)

process.TestBeginProcessBlockFilRead = cms.EDFilter("edmtest::stream::TestBeginProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockFil", "begin")
)

process.TestEndProcessBlockFil = cms.EDFilter("edmtest::stream::TestEndProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("")
)

process.TestEndProcessBlockFilRead = cms.EDFilter("edmtest::stream::TestEndProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockFil", "end")
)

process.TestBeginRunFil = cms.EDFilter("edmtest::stream::TestBeginRunFilter",
    transitions = cms.int32(nEvt+3*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndRunFil = cms.EDFilter("edmtest::stream::TestEndRunFilter",
    transitions = cms.int32(nEvt+3*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestBeginLumiBlockFil = cms.EDFilter("edmtest::stream::TestBeginLumiBlockFilter",
    transitions = cms.int32(nEvt+3*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndLumiBlockFil = cms.EDFilter("edmtest::stream::TestEndLumiBlockFilter",
    transitions = cms.int32(nEvt+3*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)

process.TestAccumulator1 = cms.EDProducer("edmtest::stream::TestAccumulator",
  expectedCount = cms.uint32(512)
)

process.TestAccumulator2 = cms.EDProducer("edmtest::stream::TestAccumulator",
  expectedCount = cms.uint32(35)
)

process.testFilterModule = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(5),
  onlyOne = cms.untracked.bool(False)
)

process.task = cms.Task(process.TestAccumulator1)

process.p = cms.Path(process.GlobIntProd +
                     process.RunIntProd +
                     process.LumiIntProd +
                     process.RunSumIntProd +
                     process.LumiSumIntProd +
                     process.ProcessBlockIntProd +
                     process.TestBeginProcessBlockProdRead +
                     process.TestBeginProcessBlockProd +
                     process.TestEndProcessBlockProdRead +
                     process.TestEndProcessBlockProd +
                     process.ProcessBlockIntProdNoGlobalCache +
                     process.TestBeginProcessBlockProdNoGlobalCache +
                     process.TestEndProcessBlockProdNoGlobalCache +
                     process.TestBeginRunProd +
                     process.TestEndRunProd +
                     process.TestBeginLumiBlockProd +
                     process.TestEndLumiBlockProd +
                     process.GlobIntAn +
                     process.RunIntAn +
                     process.LumiIntAn +
                     process.RunSumIntAn +
                     process.LumiSumIntAn +
                     process.ProcessBlockIntAn +
                     process.GlobIntFil +
                     process.RunIntFil +
                     process.LumiIntFil +
                     process.RunSumIntFil +
                     process.LumiSumIntFil +
                     process.ProcessBlockIntFil +
                     process.TestBeginProcessBlockFil +
                     process.TestBeginProcessBlockFilRead +
                     process.TestEndProcessBlockFil +
                     process.TestEndProcessBlockFilRead +
                     process.TestBeginRunFil +
                     process.TestEndRunFil +
                     process.TestBeginLumiBlockFil +
                     process.TestEndLumiBlockFil +
                     process.testFilterModule +
                     process.TestAccumulator2,
                     process.task)
