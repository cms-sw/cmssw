import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(1220227200),
    lastValue = cms.uint64(1220227200),
    interval = cms.uint64(1)
)

process.demo = cms.EDAnalyzer('RiovTest')


process.p = cms.Path(process.demo)
