import FWCore.ParameterSet.Config as cms

nEvtLumi = 4
nEvtRun = 2*nEvtLumi
nStreams = 16
nEvt = nStreams*nEvtRun*nEvtLumi

process = cms.Process("TESTONEMODULES")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(nStreams)
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
    transitions = cms.int32(nEvt+2*(nEvt/nEvtRun))
)

process.WatchLumiBlockProd = cms.EDProducer("edmtest::one::WatchLumiBlocksProducer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtLumi))
)

process.BeginRunProd = cms.EDProducer("edmtest::one::TestBeginRunProducer",
    transitions = cms.int32(nEvt+(nEvt/nEvtRun))
)

process.BeginLumiBlockProd = cms.EDProducer("edmtest::one::TestBeginLumiBlockProducer",
    transitions = cms.int32(nEvt+(nEvt/nEvtLumi))
)

process.EndRunProd = cms.EDProducer("edmtest::one::TestEndRunProducer",
    transitions = cms.int32(nEvt+(nEvt/nEvtRun))
)

process.EndLumiBlockProd = cms.EDProducer("edmtest::one::TestEndLumiBlockProducer",
    transitions = cms.int32(nEvt+(nEvt/nEvtLumi))
)


process.SharedResAn = cms.EDAnalyzer("edmtest::one::SharedResourcesAnalyzer",
    transitions = cms.int32(nEvt)
)

process.WatchRunAn = cms.EDAnalyzer("edmtest::one::WatchRunsAnalyzer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtRun))
)

process.WatchLumiBlockAn = cms.EDAnalyzer("edmtest::one::WatchLumiBlocksAnalyzer",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtLumi))
)

process.SharedResFil = cms.EDFilter("edmtest::one::SharedResourcesFilter",
    transitions = cms.int32(nEvt)
)

process.WatchRunFil = cms.EDFilter("edmtest::one::WatchRunsFilter",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtRun))
)

process.WatchLumiBlockFil = cms.EDFilter("edmtest::one::WatchLumiBlocksFilter",
    transitions = cms.int32(nEvt+2*(nEvt/nEvtLumi))
)

process.BeginRunFil = cms.EDFilter("edmtest::one::BeginRunFilter",
    transitions = cms.int32(nEvt+(nEvt/nEvtRun))
)

process.BeginLumiBlockFil = cms.EDFilter("edmtest::one::BeginLumiBlockFilter",
    transitions = cms.int32(nEvt+(nEvt/nEvtLumi))
)

process.EndRunFil = cms.EDFilter("edmtest::one::EndRunFilter",
    transitions = cms.int32(nEvt+(nEvt/nEvtRun))
)

process.EndLumiBlockFil = cms.EDFilter("edmtest::one::EndLumiBlockFilter",
    transitions = cms.int32(nEvt+(nEvt/nEvtLumi))
)


process.p = cms.Path(process.SharedResProd+process.WatchRunProd+process.WatchLumiBlockProd+process.BeginRunProd+process.BeginLumiBlockProd+process.EndRunProd+process.EndLumiBlockProd+process.SharedResAn+process.WatchRunAn+process.WatchLumiBlockAn+process.SharedResFil+process.WatchRunFil+process.WatchLumiBlockFil+process.BeginRunFil+process.BeginLumiBlockFil+process.EndRunFil+process.EndLumiBlockFil)


