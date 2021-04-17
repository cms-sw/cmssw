import FWCore.ParameterSet.Config as cms

nEvtLumi = 4
nEvtRun = 2*nEvtLumi
nRuns = 64
nStreams = 4
nEvt = nRuns*nEvtRun

process = cms.Process("TESTONEMODULES")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(nStreams),
    numberOfThreads = cms.untracked.uint32(nStreams)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nEvt)
)


process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint64(10),
    firstTime = cms.untracked.uint64(1000000),
    numberEventsInRun = cms.untracked.uint32(nEvtRun),
    numberEventsInLuminosityBlock = cms.untracked.uint32(nEvtLumi)
)

#process.Tracer = cms.Service("Tracer")

process.SharedResProd = cms.EDProducer("edmtest::one::SharedResourcesProducer",
    transitions = cms.int32(nEvt)
)

process.WatchRunProd = cms.EDProducer("edmtest::one::WatchRunsProducer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
)

process.WatchLumiBlockProd = cms.EDProducer("edmtest::one::WatchLumiBlocksProducer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
)

process.RunCacheProd = cms.EDProducer("edmtest::one::RunCacheProducer",
                                    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
                                    )

process.LumiBlockCacheProd = cms.EDProducer("edmtest::one::LumiBlockCacheProducer",
                                          transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
                                          )

process.ProcessBlockIntProd = cms.EDProducer("edmtest::one::ProcessBlockIntProducer",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.TestBeginProcessBlockProd = cms.EDProducer("edmtest::one::TestBeginProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("")
)

process.TestBeginProcessBlockProdRead = cms.EDProducer("edmtest::one::TestBeginProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin")
)

process.TestEndProcessBlockProd = cms.EDProducer("edmtest::one::TestEndProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("")
)

process.TestEndProcessBlockProdRead = cms.EDProducer("edmtest::one::TestEndProcessBlockProducer",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.BeginRunProd = cms.EDProducer("edmtest::one::TestBeginRunProducer",
    transitions = cms.int32(nEvt+int(nEvt/nEvtRun))
)

process.BeginLumiBlockProd = cms.EDProducer("edmtest::one::TestBeginLumiBlockProducer",
    transitions = cms.int32(nEvt+int(nEvt/nEvtLumi))
)

process.EndRunProd = cms.EDProducer("edmtest::one::TestEndRunProducer",
    transitions = cms.int32(nEvt+int(nEvt/nEvtRun))
)

process.EndLumiBlockProd = cms.EDProducer("edmtest::one::TestEndLumiBlockProducer",
    transitions = cms.int32(nEvt+int(nEvt/nEvtLumi))
)


process.SharedResAn = cms.EDAnalyzer("edmtest::one::SharedResourcesAnalyzer",
    transitions = cms.int32(nEvt)
)

process.WatchRunAn = cms.EDAnalyzer("edmtest::one::WatchRunsAnalyzer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
)

process.WatchLumiBlockAn = cms.EDAnalyzer("edmtest::one::WatchLumiBlocksAnalyzer",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
    # needed to avoid deleting TestAccumulator1
    ,moduleLabel = cms.InputTag("TestAccumulator1")
)

process.RunCacheAn = cms.EDAnalyzer("edmtest::one::RunCacheAnalyzer",
                                    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
                                    )

process.LumiBlockCacheAn = cms.EDAnalyzer("edmtest::one::LumiBlockCacheAnalyzer",
                                          transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
                                          )

process.ProcessBlockIntAn = cms.EDAnalyzer("edmtest::one::ProcessBlockIntAnalyzer",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockProd" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockProd", "end")
)

process.SharedResFil = cms.EDFilter("edmtest::one::SharedResourcesFilter",
    transitions = cms.int32(nEvt)
)

process.WatchRunFil = cms.EDFilter("edmtest::one::WatchRunsFilter",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
)

process.WatchLumiBlockFil = cms.EDFilter("edmtest::one::WatchLumiBlocksFilter",
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
)
process.RunCacheFil = cms.EDFilter("edmtest::one::RunCacheFilter",
                                    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
                                    )

process.LumiBlockCacheFil = cms.EDFilter("edmtest::one::LumiBlockCacheFilter",
                                          transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
                                          )

process.ProcessBlockIntFil = cms.EDFilter("edmtest::one::ProcessBlockIntFilter",
    transitions = cms.int32(int(nEvt + 2)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockFil" ,"begin"),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockFil", "end")
)

process.TestBeginProcessBlockFil = cms.EDFilter("edmtest::one::TestBeginProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("")
)

process.TestBeginProcessBlockFilRead = cms.EDFilter("edmtest::one::TestBeginProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesBeginProcessBlock = cms.InputTag("TestBeginProcessBlockFil" ,"begin")
)

process.TestEndProcessBlockFil = cms.EDFilter("edmtest::one::TestEndProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("")
)

process.TestEndProcessBlockFilRead = cms.EDFilter("edmtest::one::TestEndProcessBlockFilter",
    transitions = cms.int32(int(nEvt + 1)),
    consumesEndProcessBlock = cms.InputTag("TestEndProcessBlockFil", "end")
)

process.BeginRunFil = cms.EDFilter("edmtest::one::BeginRunFilter",
    transitions = cms.int32(nEvt+int(nEvt/nEvtRun))
)

process.BeginLumiBlockFil = cms.EDFilter("edmtest::one::BeginLumiBlockFilter",
    transitions = cms.int32(nEvt+int(nEvt/nEvtLumi))
)

process.EndRunFil = cms.EDFilter("edmtest::one::EndRunFilter",
    transitions = cms.int32(nEvt+int(nEvt/nEvtRun))
)

process.EndLumiBlockFil = cms.EDFilter("edmtest::one::EndLumiBlockFilter",
    transitions = cms.int32(nEvt+int(nEvt/nEvtLumi))
)

process.TestAccumulator1 = cms.EDProducer("edmtest::one::TestAccumulator",
  expectedCount = cms.uint32(512)
)

process.TestAccumulator2 = cms.EDProducer("edmtest::one::TestAccumulator",
  expectedCount = cms.uint32(35)
)

process.testFilterModule = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(5),
  onlyOne = cms.untracked.bool(False)
)

process.task = cms.Task(process.TestAccumulator1)


process.p = cms.Path(process.SharedResProd +
                     process.WatchRunProd +
                     process.WatchLumiBlockProd +
                     process.RunCacheProd +
                     process.LumiBlockCacheProd +
                     process.ProcessBlockIntProd +
                     process.TestBeginProcessBlockProd +
                     process.TestBeginProcessBlockProdRead +
                     process.TestEndProcessBlockProd +
                     process.TestEndProcessBlockProdRead +
                     process.BeginRunProd +
                     process.BeginLumiBlockProd +
                     process.EndRunProd +
                     process.EndLumiBlockProd +
                     process.SharedResAn +
                     process.WatchRunAn +
                     process.WatchLumiBlockAn +
                     process.RunCacheAn +
                     process.LumiBlockCacheAn +
                     process.ProcessBlockIntAn +
                     process.SharedResFil +
                     process.WatchRunFil +
                     process.WatchLumiBlockFil +
                     process.RunCacheFil +
                     process.LumiBlockCacheFil +
                     process.ProcessBlockIntFil +
                     process.TestBeginProcessBlockFil +
                     process.TestBeginProcessBlockFilRead +
                     process.TestEndProcessBlockFil +
                     process.TestEndProcessBlockFilRead +
                     process.BeginRunFil +
                     process.BeginLumiBlockFil +
                     process.EndRunFil +
                     process.EndLumiBlockFil +
                     process.testFilterModule +
                     process.TestAccumulator2,
                     process.task)
