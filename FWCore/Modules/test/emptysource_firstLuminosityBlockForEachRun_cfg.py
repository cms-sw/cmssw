# Configuration file for EmptySource

from builtins import range
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(8*5)
)

runToLumi = ((2,1),(10,3),(20,7) )

def findRunForLumi( lumi) :
  lastRun = runToLumi[0][0]
  for r,l in runToLumi:
    if l > lumi:
      break
    lastRun = r
  return lastRun

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(*[cms.LuminosityBlockID(x,y) for x,y in runToLumi]),
    numberEventsInLuminosityBlock = cms.untracked.uint32(5),
    firstTime = cms.untracked.uint64(1000),
    timeBetweenEvents = cms.untracked.uint64(10)
)

ids = cms.VEventID()
numberOfEventsInLumi = 0
numberOfEventsPerLumi = process.source.numberEventsInLuminosityBlock.value()
lumi = process.source.firstLuminosityBlock.value()
event=0
oldRun = 2
for i in range(process.maxEvents.input.value()):
   numberOfEventsInLumi +=1
   event += 1
   run = findRunForLumi(lumi)
   if numberOfEventsInLumi > numberOfEventsPerLumi:
      numberOfEventsInLumi=1
      lumi += 1
      run = findRunForLumi(lumi)
      if run != oldRun:
        event = 1
        oldRun = run
   ids.append(cms.EventID(run,lumi,event))
process.check = cms.EDAnalyzer("EventIDChecker", eventSequence = cms.untracked(ids))
process.print1 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.check+process.print1)
