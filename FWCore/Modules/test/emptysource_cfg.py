# Configuration file for EmptySource

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(100),
    numberEventsInRun = cms.untracked.uint32(5),
    firstTime = cms.untracked.uint64(1000),
    timeBetweenEvents = cms.untracked.uint64(10)
)

ids = cms.VEventID()
numberOfEventsInRun = 0
numberOfEventsPerRun = process.source.numberEventsInRun.value()
run = process.source.firstRun.value()
event=0
for i in xrange(process.maxEvents.input.value()):
   numberOfEventsInRun +=1
   event += 1
   if numberOfEventsInRun > numberOfEventsPerRun:
      numberOfEventsInRun=1
      run += 1
      event = 1
   ids.append(cms.EventID(run,event))
process.check = cms.EDAnalyzer("EventIDChecker", eventSequence = cms.untracked(ids))
process.print1 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.check+process.print1)
