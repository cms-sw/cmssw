# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int(sys.argv[2]))
)

firstRun = int(sys.argv[3])
numberEventsInRun = int(sys.argv[4])
firstLuminosityBlock = int(sys.argv[5])
numberEventsInLuminosityBlock = int(sys.argv[6])

if len(sys.argv) > 7:
  #have first run of second file continue the run
  runToLumi = ((2,1),(10,3),(20,5), (30, 7), (40, 9) )
else:
  #have first run of second file be a new run
  runToLumi = ((2,1),(10,3),(20,5), (30, 6), (40, 9) )

def findRunForLumi( lumi) :
  lastRun = runToLumi[0][0]
  for r,l in runToLumi:
    if l > lumi:
      break
    lastRun = r
  return lastRun

ids = cms.VEventID()
numberOfEventsInLumi = 0
numberOfEventsPerLumi = numberEventsInLuminosityBlock
lumi = 1
event=0
oldRun = 2
numberOfFiles = len(sys.argv[1].split(","))
numberOfEventsInFile = int(sys.argv[2])/numberOfFiles
for i in range(process.maxEvents.input.value()):
   numberOfEventsInLumi +=1
   event += 1
   run = findRunForLumi(lumi)
   if event > numberOfEventsInFile:
    event = 1
   if numberOfEventsInLumi > numberOfEventsPerLumi:
      numberOfEventsInLumi=1
      lumi += 1
      run = findRunForLumi(lumi)
      if run != oldRun:
        oldRun = run
   ids.append(cms.EventID(run,lumi,event))
process.check = cms.EDAnalyzer("EventIDChecker", eventSequence = cms.untracked(ids))


process.source = cms.Source("PoolSource",
    firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(*[cms.LuminosityBlockID(x,y) for x,y in runToLumi]),
    fileNames = cms.untracked.vstring(sys.argv[1].split(","))
)

process.e = cms.EndPath(process.check)

