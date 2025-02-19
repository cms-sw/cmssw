#sometime in laste 1980
start = 10*365*3600*24
end = start+100
import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(start<<32),
    lastValue = cms.uint64(end<<32),
    interval = cms.uint64(10<<32)
)



