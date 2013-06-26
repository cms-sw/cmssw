import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('lumiid'),
    firstValue = cms.uint64((1211<<32)+1),
    lastValue = cms.uint64((1211<<32)+8),
    interval = cms.uint64(2)
)



