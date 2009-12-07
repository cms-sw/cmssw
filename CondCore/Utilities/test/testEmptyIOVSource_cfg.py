import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(8),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(2),
    interval = cms.uint64(2)
)



