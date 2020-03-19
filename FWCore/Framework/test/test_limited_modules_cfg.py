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

#process.Tracer = cms.Service("Tracer")

process.StreamIntProd = cms.EDProducer("edmtest::limited::StreamIntProducer",
    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(2*int(nEvt/nEvtRun)+2*int(nEvt/nEvtLumi)+2))
    ,cachevalue = cms.int32(1)
)

process.RunIntProd = cms.EDProducer("edmtest::limited::RunIntProducer",
                                    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntProd = cms.EDProducer("edmtest::limited::LumiIntProducer",
                                     concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntProd = cms.EDProducer("edmtest::limited::RunSummaryIntProducer",
                                       concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams*int(nEvt/nEvtRun)+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntProd = cms.EDProducer("edmtest::limited::LumiSummaryIntProducer",
                                        concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nStreams*int(nEvt/nEvtLumi)+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.TestBeginRunProd = cms.EDProducer("edmtest::limited::TestBeginRunProducer",
                                          concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestEndRunProd = cms.EDProducer("edmtest::limited::TestEndRunProducer",
                                        concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestBeginLumiBlockProd = cms.EDProducer("edmtest::limited::TestBeginLumiBlockProducer",
                                                concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.TestEndLumiBlockProd = cms.EDProducer("edmtest::limited::TestEndLumiBlockProducer",
                                              concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.StreamIntAn = cms.EDAnalyzer("edmtest::limited::StreamIntAnalyzer",
                                     concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(2*int(nEvt/nEvtRun)+2*int(nEvt/nEvtLumi)+2))
    ,cachevalue = cms.int32(1)
)

process.RunIntAn= cms.EDAnalyzer("edmtest::limited::RunIntAnalyzer",
                                 concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntAn = cms.EDAnalyzer("edmtest::limited::LumiIntAnalyzer",
                                   concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntAn = cms.EDAnalyzer("edmtest::limited::RunSummaryIntAnalyzer",
                                     concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(int(nEvt/nEvtRun)+1)+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntAn = cms.EDAnalyzer("edmtest::limited::LumiSummaryIntAnalyzer",
                                      concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(int(nEvt/nEvtLumi)+1)+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.StreamIntFil = cms.EDFilter("edmtest::limited::StreamIntFilter",
                                    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(2*int(nEvt/nEvtRun)+2*int(nEvt/nEvtLumi)+2))
    ,cachevalue = cms.int32(1)
)

process.RunIntFil = cms.EDFilter("edmtest::limited::RunIntFilter",
                                 concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntFil = cms.EDFilter("edmtest::limited::LumiIntFilter",
                                  concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntFil = cms.EDFilter("edmtest::limited::RunSummaryIntFilter",
                                    concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(int(nEvt/nEvtRun)+1)+2*int(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntFil = cms.EDFilter("edmtest::limited::LumiSummaryIntFilter",
                                     concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(nEvt+nStreams*(int(nEvt/nEvtLumi)+1)+2*int(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.TestBeginRunFil = cms.EDFilter("edmtest::limited::TestBeginRunFilter",
                                       concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestEndRunFil = cms.EDFilter("edmtest::limited::TestEndRunFilter",
                                     concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtRun))
)

process.TestBeginLumiBlockFil = cms.EDFilter("edmtest::limited::TestBeginLumiBlockFilter",
                                             concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.TestEndLumiBlockFil = cms.EDFilter("edmtest::limited::TestEndLumiBlockFilter",
                                           concurrencyLimit = cms.untracked.uint32(1),
    transitions = cms.int32(int(nEvt/nEvtLumi))
)

process.TestAccumulator1 = cms.EDProducer("edmtest::limited::TestAccumulator",
  expectedCount = cms.uint32(512),
  concurrencyLimit = cms.untracked.uint32(2)
)

process.TestAccumulator2 = cms.EDProducer("edmtest::limited::TestAccumulator",
  expectedCount = cms.uint32(35),
  concurrencyLimit = cms.untracked.uint32(2)
)

process.testFilterModule = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(5),
  onlyOne = cms.untracked.bool(False)
)

process.task = cms.Task(process.TestAccumulator1)


process.p = cms.Path(process.StreamIntProd+process.RunIntProd+process.LumiIntProd+process.RunSumIntProd+process.LumiSumIntProd+process.TestBeginRunProd+process.TestEndRunProd+process.TestBeginLumiBlockProd+process.TestEndLumiBlockProd+process.StreamIntAn+process.RunIntAn+process.LumiIntAn+process.RunSumIntAn+process.LumiSumIntAn+process.StreamIntFil+process.RunIntFil+process.LumiIntFil+process.RunSumIntFil+process.LumiSumIntFil+process.TestBeginRunFil+process.TestEndRunFil+process.TestBeginLumiBlockFil+process.TestEndLumiBlockFil+process.testFilterModule+process.TestAccumulator2, process.task)
