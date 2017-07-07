# Configuration file for EmptySource

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.options = cms.untracked.PSet(multiProcesses=cms.untracked.PSet(
        maxChildProcesses=cms.untracked.int32(3),
        maxSequentialEventsPerChild=cms.untracked.uint32(2)))


process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring("file:multiprocess_oneRun_test.root"),
skipEvents = cms.untracked.uint32(6))

ids = cms.VEventID()
numberOfEventsInRun = 0
numberOfEventsPerRun = 100
run = 100
event=6
for i in xrange(20):
   numberOfEventsInRun +=1
   event += 1
   if numberOfEventsInRun > numberOfEventsPerRun:
      numberOfEventsInRun=1
      run += 1
      event = 1
   ids.append(cms.EventID(run,event))
process.check = cms.EDAnalyzer("EventIDChecker", 
                                eventSequence = cms.untracked(ids),
                                multiProcessSequentialEvents = process.options.multiProcesses.maxSequentialEventsPerChild)
process.print1 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.check+process.print1)
