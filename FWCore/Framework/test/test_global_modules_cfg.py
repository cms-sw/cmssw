import FWCore.ParameterSet.Config as cms

nEvtLumi = 4
nEvtRun = 2*nEvtLumi
nRuns = 64
nStreams = 4
nEvt = nRuns*nEvtRun

process = cms.Process("TESTGLOBALMODULES")

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

process.StreamIntProd = cms.EDProducer("edmtest::global::StreamIntProducer",
    transitions = cms.int32(int(nEvt+nStreams*(2*(nEvt/nEvtRun)+2*(nEvt/nEvtLumi)+2)))
    ,cachevalue = cms.int32(1)
)

process.RunIntProd = cms.EDProducer("edmtest::global::RunIntProducer",
    transitions = cms.int32(int(2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntProd = cms.EDProducer("edmtest::global::LumiIntProducer",
    transitions = cms.int32(int(2*(nEvt/nEvtLumi)))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntProd = cms.EDProducer("edmtest::global::RunSummaryIntProducer",
    transitions = cms.int32(int(nStreams*(nEvt/nEvtRun)+2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntProd = cms.EDProducer("edmtest::global::LumiSummaryIntProducer",
    transitions = cms.int32(int(nStreams*(nEvt/nEvtLumi)+2*(nEvt/nEvtLumi)))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.ProcessBlockIntProd = cms.EDProducer("edmtest::global::ProcessBlockIntProducer",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.TestBeginProcessBlockProd = cms.EDProducer("edmtest::global::TestBeginProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("")
)

process.TestBeginProcessBlockProdRead = cms.EDProducer("edmtest::global::TestBeginProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin")
)

process.TestEndProcessBlockProd = cms.EDProducer("edmtest::global::TestEndProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("")
)

process.TestEndProcessBlockProdRead = cms.EDProducer("edmtest::global::TestEndProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.TestBeginRunProd = cms.EDProducer("edmtest::global::TestBeginRunProducer",
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestEndRunProd = cms.EDProducer("edmtest::global::TestEndRunProducer",
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestBeginLumiBlockProd = cms.EDProducer("edmtest::global::TestBeginLumiBlockProducer",
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.TestEndLumiBlockProd = cms.EDProducer("edmtest::global::TestEndLumiBlockProducer",
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.StreamIntAn = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer",
    transitions = cms.int32(int(nEvt+nStreams*(2*(nEvt/nEvtRun)+2*(nEvt/nEvtLumi)+2)))
    ,cachevalue = cms.int32(1)
)

process.RunIntAn= cms.EDAnalyzer("edmtest::global::RunIntAnalyzer",
    transitions = cms.int32(int(nEvt+2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntAn = cms.EDAnalyzer("edmtest::global::LumiIntAnalyzer",
    transitions = cms.int32(int(nEvt+2*(nEvt/nEvtLumi)))
    ,cachevalue = cms.int32(nEvtLumi)
    # needed to avoid deleting TestAccumulator1
    ,moduleLabel = cms.InputTag("TestAccumulator1")
)

process.RunSumIntAn = cms.EDAnalyzer("edmtest::global::RunSummaryIntAnalyzer",
    transitions = cms.int32(int(nEvt+nStreams*((nEvt/nEvtRun)+1)+2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntAn = cms.EDAnalyzer("edmtest::global::LumiSummaryIntAnalyzer",
    transitions = cms.int32(int(nEvt+nStreams*((nEvt/nEvtLumi)+1)+2*(nEvt/nEvtLumi)))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.ProcessBlockIntAn = cms.EDAnalyzer("edmtest::global::ProcessBlockIntAnalyzer",
    transitions = cms.int32(652),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.StreamIntFil = cms.EDFilter("edmtest::global::StreamIntFilter",
    transitions = cms.int32(int(nEvt+nStreams*(2*(nEvt/nEvtRun)+2*(nEvt/nEvtLumi)+2)))
    ,cachevalue = cms.int32(1)
)

process.RunIntFil = cms.EDFilter("edmtest::global::RunIntFilter",
    transitions = cms.int32(int(nEvt+2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntFil = cms.EDFilter("edmtest::global::LumiIntFilter",
    transitions = cms.int32(int(nEvt+2*(nEvt/nEvtLumi)))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntFil = cms.EDFilter("edmtest::global::RunSummaryIntFilter",
    transitions = cms.int32(int(nEvt+nStreams*((nEvt/nEvtRun)+1)+2*(nEvt/nEvtRun)))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntFil = cms.EDFilter("edmtest::global::LumiSummaryIntFilter",
    transitions = cms.int32(int(nEvt+nStreams*((nEvt/nEvtLumi)+1)+2*(nEvt/nEvtLumi)))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.ProcessBlockIntFil = cms.EDFilter("edmtest::global::ProcessBlockIntFilter",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockFil" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockFil", "end")
)

process.TestBeginProcessBlockFil = cms.EDFilter("edmtest::global::TestBeginProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("")
)

process.TestBeginProcessBlockFilRead = cms.EDFilter("edmtest::global::TestBeginProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockFil" ,"begin")
)

process.TestEndProcessBlockFil = cms.EDFilter("edmtest::global::TestEndProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("")
)

process.TestEndProcessBlockFilRead = cms.EDFilter("edmtest::global::TestEndProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockFil", "end")
)

process.TestBeginRunFil = cms.EDFilter("edmtest::global::TestBeginRunFilter",
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestEndRunFil = cms.EDFilter("edmtest::global::TestEndRunFilter",
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestBeginLumiBlockFil = cms.EDFilter("edmtest::global::TestBeginLumiBlockFilter",
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.TestEndLumiBlockFil = cms.EDFilter("edmtest::global::TestEndLumiBlockFilter",
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.TestAccumulator1 = cms.EDProducer("edmtest::global::TestAccumulator",
  expectedCount = cms.uint32(512)
)

process.TestAccumulator2 = cms.EDProducer("edmtest::global::TestAccumulator",
  expectedCount = cms.uint32(35)
)

process.testFilterModule = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(5),
  onlyOne = cms.untracked.bool(False)
)

process.task = cms.Task(process.TestAccumulator1)


process.p = cms.Path(process.StreamIntProd +
                     process.RunIntProd +
                     process.LumiIntProd +
                     process.RunSumIntProd +
                     process.LumiSumIntProd +
                     process.ProcessBlockIntProd +
                     process.TestBeginProcessBlockProdRead +
                     process.TestBeginProcessBlockProd +
                     process.TestEndProcessBlockProdRead +
                     process.TestEndProcessBlockProd +
                     process.TestBeginRunProd +
                     process.TestEndRunProd +
                     process.TestBeginLumiBlockProd +
                     process.TestEndLumiBlockProd +
                     process.StreamIntAn +
                     process.RunIntAn +
                     process.LumiIntAn +
                     process.RunSumIntAn +
                     process.LumiSumIntAn +
                     process.ProcessBlockIntAn +
                     process.StreamIntFil +
                     process.RunIntFil +
                     process.LumiIntFil +
                     process.RunSumIntFil +
                     process.LumiSumIntFil +
                     process.ProcessBlockIntFil +
                     process.TestBeginProcessBlockFilRead +
                     process.TestBeginProcessBlockFil +
                     process.TestEndProcessBlockFilRead +
                     process.TestEndProcessBlockFil +
                     process.TestBeginRunFil +
                     process.TestEndRunFil +
                     process.TestBeginLumiBlockFil +
                     process.TestEndLumiBlockFil +
                     process.testFilterModule +
                     process.TestAccumulator2,
                     process.task)
