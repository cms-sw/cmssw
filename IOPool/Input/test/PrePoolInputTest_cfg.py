# The following comments couldn't be translated into the new config version:

# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms
import sys

useOtherThing = False
if len(sys.argv) > 7:
  if sys.argv[7] == "useOtherThing":
    useOtherThing = True

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents.input = int(sys.argv[2])

process.Thing = cms.EDProducer("ThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(sys.argv[1])
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(int(sys.argv[3])),
    numberEventsInRun = cms.untracked.uint32(int(sys.argv[4])),
    firstLuminosityBlock = cms.untracked.uint32(int(sys.argv[5])),
    numberEventsInLuminosityBlock = cms.untracked.uint32(int(sys.argv[6]))
)

process.p = cms.Path(process.Thing)
if useOtherThing:
  process.OtherThing = cms.EDProducer("OtherThingProducer")
  process.p = cms.Path(process.Thing + process.OtherThing)
process.ep = cms.EndPath(process.output)


