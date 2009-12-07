import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(4294967298),
    lastValue = cms.uint64(4294967302),
    interval = cms.uint64(1)
)



