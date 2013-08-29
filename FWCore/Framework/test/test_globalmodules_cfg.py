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

process.SIP = cms.EDProducer("StreamIntGProducer",
    transitions = cms.int32(7)
)

process.RIP = cms.EDProducer("RunIntGProducer",
    transitions = cms.int32(3)
)

process.LIP = cms.EDProducer("LumiIntGProducer",
    transitions = cms.int32(3)
)

process.RSIP = cms.EDProducer("RunSummaryIntGProducer",
    transitions = cms.int32(4)
)

process.LSIP = cms.EDProducer("LumiSummaryIntGProducer",
    transitions = cms.int32(4)
)
process.SIA = cms.EDAnalyzer("StreamIntGAnalzer",
    transitions = cms.int32(7)
)

process.RIA = cms.EDAnalyzer("RunIntGAnalzer",
    transitions = cms.int32(3)
)

process.LIA = cms.EDAnalyzer("LumiIntGAnalzer",
    transitions = cms.int32(3)
)

process.RSIA = cms.EDAnalyzer("RunSummaryIntGAnalzer",
    transitions = cms.int32(4)
)

process.LSIA = cms.EDAnalyzer("LumiSummaryIntGAnalzer",
    transitions = cms.int32(4)
)

process.SIF = cms.EDFilter("StreamIntGFilter",
    transitions = cms.int32(7)
)

process.RIF = cms.EDFilter("RunIntGFilter",
    transitions = cms.int32(3)
)

process.LIF = cms.EDFilter("LumiIntGFilter",
    transitions = cms.int32(3)
)

process.RSIF = cms.EDFilter("RunSummaryIntGFilter",
    transitions = cms.int32(4)
)

process.LSIF = cms.EDFilter("LumiSummaryIntGFilter",
    transitions = cms.int32(4)
)


process.p = cms.Path(process.SIP+process.RIP+process.LIP+process.RSIP+process.LSIP+process.SIA+process.RIA+process.LIA+process.RSIA+process.LSIA+process.SIF+process.RIF+process.LIF+process.RSIF+process.LSIF)


