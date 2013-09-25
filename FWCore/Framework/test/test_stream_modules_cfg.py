import FWCore.ParameterSet.Config as cms

nEvtLumi = 4
nEvtRun = 2*nEvtLumi
nStreams = 64
nEvt = (nStreams/4)*nEvtRun*nEvtLumi

process = cms.Process("TESTSTREAMMODULES")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(nStreams)
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
    transitions = cms.int32(nEvt+2)
    ,cachevalue = cms.int32(nEvt)
)

process.RunIntProd = cms.EDProducer("edmtest::stream::RunIntProducer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntProd = cms.EDProducer("edmtest::stream::LumiIntProducer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntProd = cms.EDProducer("edmtest::stream::RunSummaryIntProducer",
    transitions = cms.int32(nEvt+4*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntProd = cms.EDProducer("edmtest::stream::LumiSummaryIntProducer",
    transitions = cms.int32(nEvt+4*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.TestBeginRunProd = cms.EDProducer("edmtest::stream::TestBeginRunProducer",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndRunProd = cms.EDProducer("edmtest::stream::TestEndRunProducer",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestBeginLumiBlockProd = cms.EDProducer("edmtest::stream::TestBeginLumiBlockProducer",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndLumiBlockProd = cms.EDProducer("edmtest::stream::TestEndLumiBlockProducer",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)


process.GlobIntAn = cms.EDAnalyzer("edmtest::stream::GlobalIntAnalyzer",
    transitions = cms.int32(nEvt+2)
    ,cachevalue = cms.int32(nEvt)
)

process.RunIntAn= cms.EDAnalyzer("edmtest::stream::RunIntAnalyzer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntAn = cms.EDAnalyzer("edmtest::stream::LumiIntAnalyzer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntAn = cms.EDAnalyzer("edmtest::stream::RunSummaryIntAnalyzer",
    transitions = cms.int32(nEvt+4*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntAn = cms.EDAnalyzer("edmtest::stream::LumiSummaryIntAnalyzer",
    transitions = cms.int32(nEvt+4*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.GlobIntFil = cms.EDFilter("edmtest::stream::GlobalIntFilter",
    transitions = cms.int32(nEvt+2)
    ,cachevalue = cms.int32(nEvt)
)

process.RunIntFil = cms.EDFilter("edmtest::stream::RunIntFilter",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiIntFil = cms.EDFilter("edmtest::stream::LumiIntFilter",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)

process.RunSumIntFil = cms.EDFilter("edmtest::stream::RunSummaryIntFilter",
    transitions = cms.int32(nEvt+4*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvtRun)
)

process.LumiSumIntFil = cms.EDFilter("edmtest::stream::LumiSummaryIntFilter",
    transitions = cms.int32(nEvt+4*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvtLumi)
)
process.TestBeginRunFil = cms.EDFilter("edmtest::stream::TestBeginRunFilter",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndRunFil = cms.EDFilter("edmtest::stream::TestEndRunFilter",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtRun))
    ,cachevalue = cms.int32(nEvt)
)

process.TestBeginLumiBlockFil = cms.EDFilter("edmtest::stream::TestBeginLumiBlockFilter",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)

process.TestEndLumiBlockFil = cms.EDFilter("edmtest::stream::TestEndLumiBlockFilter",
    transitions = cms.int32(nEvt+3*(nEvt/nEvtLumi))
    ,cachevalue = cms.int32(nEvt)
)


process.p = cms.Path(process.GlobIntProd+process.RunIntProd+process.LumiIntProd+process.RunSumIntProd+process.LumiSumIntProd+process.TestBeginRunProd+process.TestEndRunProd+process.TestBeginLumiBlockProd+process.TestEndLumiBlockProd+process.GlobIntAn+process.RunIntAn+process.LumiIntAn+process.RunSumIntAn+process.LumiSumIntAn+process.GlobIntFil+process.RunIntFil+process.LumiIntFil+process.RunSumIntFil+process.LumiSumIntFil+process.TestBeginRunFil+process.TestEndRunFil+process.TestBeginLumiBlockFil+process.TestEndLumiBlockFil)

