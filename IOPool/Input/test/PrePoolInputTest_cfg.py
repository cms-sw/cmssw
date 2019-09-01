# The following comments couldn't be translated into the new config version:

# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms
from sys import argv

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int(argv[3]))
)

process.Thing = cms.EDProducer("ThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(argv[2])
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(int(argv[4])),
    numberEventsInRun = cms.untracked.uint32(int(argv[5])),
    firstLuminosityBlock = cms.untracked.uint32(int(argv[6])),
    numberEventsInLuminosityBlock = cms.untracked.uint32(int(argv[7]))
)

process.p = cms.Path(process.Thing)
process.ep = cms.EndPath(process.output)


