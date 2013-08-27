import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(False),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint64(10),
    firstTime = cms.untracked.uint64(1000000)
)

process.Tracer = cms.Service("Tracer")

process.SIP = cms.EDProducer("StreamIntProducer",
    ivalue = cms.int32(0)
)

process.RIP = cms.EDProducer("RunIntProducer",
    ivalue = cms.int32(0)
)

process.LIP = cms.EDProducer("LumiIntProducer",
    ivalue = cms.int32(0)
)

process.RSIP = cms.EDProducer("RunSummaryIntProducer",
    ivalue = cms.int32(0)
)

process.LSIP = cms.EDProducer("LumiSummaryIntProducer",
    ivalue = cms.int32(0)
)
process.SIA = cms.EDAnalyzer("StreamIntAnalyzer",
    ivalue = cms.int32(0)
)

process.RIA = cms.EDAnalyzer("RunIntAnalyzer",
    ivalue = cms.int32(0)
)

process.LIA = cms.EDAnalyzer("LumiIntAnalyzer",
    ivalue = cms.int32(0)
)

process.RSIA = cms.EDAnalyzer("RunSummaryIntAnalyzer",
    ivalue = cms.int32(0)
)

process.LSIA = cms.EDAnalyzer("LumiSummaryIntAnalyzer",
    ivalue = cms.int32(0)
)

process.SIF = cms.EDFilter("StreamIntFilter",
    ivalue = cms.int32(0)
)

process.RIF = cms.EDFilter("RunIntFilter",
    ivalue = cms.int32(0)
)

process.LIF = cms.EDFilter("LumiIntFilter",
    ivalue = cms.int32(0)
)

process.RSIF = cms.EDFilter("RunSummaryIntFilter",
    ivalue = cms.int32(0)
)

process.LSIF = cms.EDFilter("LumiSummaryIntFilter",
    ivalue = cms.int32(0)
)


process.p = cms.Path(process.SIP+process.RIP+process.LIP+process.RSIP+process.LSIP+process.SIA+process.RIA+process.LIA+process.RSIA+process.LSIA+process.SIF+process.RIF+process.LIF+process.RSIF+process.LSIF)


