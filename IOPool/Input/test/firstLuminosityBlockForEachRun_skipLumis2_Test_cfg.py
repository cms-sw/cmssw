# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents.input = -1

#have first run of second file be a new run
runToLumi = ((2,1),(10,3),(20,5))

def findRunForLumi( lumi) :
  lastRun = runToLumi[0][0]
  for r,l in runToLumi:
    if l > lumi:
      break
    lastRun = r
  return lastRun

ids = cms.VEventID()
numberOfEventsInLumi = 0
numberOfEventsPerLumi = 5
lumi = int(sys.argv[2])
event= numberOfEventsPerLumi*(lumi-1)
oldRun = 2
numberOfFiles = 1
numberOfEventsInFile = 15
for i in range(numberOfEventsPerLumi*(6-lumi)):
   numberOfEventsInLumi +=1
   event += 1
   run = findRunForLumi(lumi)
#   if event > numberOfEventsInFile:
#    event = 1
   if numberOfEventsInLumi > numberOfEventsPerLumi:
      numberOfEventsInLumi=1
      lumi += 1
      run = findRunForLumi(lumi)
      if run != oldRun:
        oldRun = run
   ids.append(cms.EventID(run,lumi,event))
process.check = cms.EDAnalyzer("EventIDChecker", eventSequence = cms.untracked(ids))


process.source = cms.Source("PoolSource",
                            lumisToProcess = cms.untracked.VLuminosityBlockRange('1:'+str(int(sys.argv[2]))+'-1:'+str(runToLumi[-1][1])),
    firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(*[cms.LuminosityBlockID(x,y) for x,y in runToLumi]),
    fileNames = cms.untracked.vstring(sys.argv[1].split(","))
)

process.e = cms.EndPath(process.check)
