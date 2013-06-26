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


process.source = cms.Source("NewEventStreamFileReader", fileNames = cms.untracked.vstring("file:multiprocess_test.dat"))

ids = cms.VEventID()
numberOfEventsInRun = 0
numberOfEventsPerRun = 500
numberOfEventsInLumi = 0
numberOfEventsPerLumi = 2
run = 100
lumi = 1
event=0
ids.append(cms.EventID(run,0,0))
ids.append(cms.EventID(run,lumi,0))
for i in xrange(20):
   numberOfEventsInRun +=1
   numberOfEventsInLumi +=1
   event += 1
   if numberOfEventsInRun > numberOfEventsPerRun:
      ids.append(cms.EventID(run,lumi,0))
      ids.append(cms.EventID(run,0,0))
      numberOfEventsInRun=1
      numberOfEventsInLumi=1
      run += 1
      event = 1
      lumi = 1
      ids.append(cms.EventID(run,0,0))
      ids.append(cms.EventID(run,lumi,0))
   if numberOfEventsInLumi > numberOfEventsPerLumi:
       numberOfEventsInLumi = 1
       ids.append(cms.EventID(run,lumi,0))
       lumi += 1
       ids.append(cms.EventID(run,lumi,0))
   ids.append(cms.EventID(run,lumi,event))
ids.append(cms.EventID(run,lumi,0))
ids.append(cms.EventID(run,0,0))

process.check = cms.EDAnalyzer("MulticoreRunLumiEventChecker", 
                                eventSequence = cms.untracked(ids),
                                multiProcessSequentialEvents = process.options.multiProcesses.maxSequentialEventsPerChild)
process.print1 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.check+process.print1)
