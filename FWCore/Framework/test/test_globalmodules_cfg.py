import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(False),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint64(10),
    firstTime = cms.untracked.uint64(1000000),
    numberEventsInRun = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1) 
)

process.Tracer = cms.Service("Tracer")

process.SIP = cms.EDProducer("edmtest::global::StreamIntProducer",
    transitions = cms.int32(7)
)

process.RIP = cms.EDProducer("edmtest::global::RunIntProducer",
    transitions = cms.int32(3)
)

process.LIP = cms.EDProducer("edmtest::global::LumiIntProducer",
    transitions = cms.int32(3)
)

process.RSIP = cms.EDProducer("edmtest::global::RunSummaryIntProducer",
    transitions = cms.int32(4)
)

process.LSIP = cms.EDProducer("edmtest::global::LumiSummaryIntProducer",
    transitions = cms.int32(4)
)
process.SIA = cms.EDAnalyzer("edmtest::global::StreamIntAnalzer",
    transitions = cms.int32(7)
)

process.RIA = cms.EDAnalyzer("edmtest::global::RunIntAnalzer",
    transitions = cms.int32(3)
)

process.LIA = cms.EDAnalyzer("edmtest::global::LumiIntAnalzer",
    transitions = cms.int32(3)
)

process.RSIA = cms.EDAnalyzer("edmtest::global::RunSummaryIntAnalzer",
    transitions = cms.int32(4)
)

process.LSIA = cms.EDAnalyzer("edmtest::global::LumiSummaryIntAnalzer",
    transitions = cms.int32(4)
)

process.SIF = cms.EDFilter("edmtest::global::StreamIntFilter",
    transitions = cms.int32(7)
)

process.RIF = cms.EDFilter("edmtest::global::RunIntFilter",
    transitions = cms.int32(3)
)

process.LIF = cms.EDFilter("edmtest::global::LumiIntFilter",
    transitions = cms.int32(3)
)

process.RSIF = cms.EDFilter("edmtest::global::RunSummaryIntFilter",
    transitions = cms.int32(4)
)

process.LSIF = cms.EDFilter("edmtest::global::LumiSummaryIntFilter",
    transitions = cms.int32(4)
)


process.p = cms.Path(process.SIP+process.RIP+process.LIP+process.RSIP+process.LSIP+process.SIA+process.RIA+process.LIA+process.RSIA+process.LSIA+process.SIF+process.RIF+process.LIF+process.RSIF+process.LSIF)


