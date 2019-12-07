# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms
import sys

argv = []
foundpy = False
for a in sys.argv:
    if foundpy:
        argv.append(a)
    if ".py" in a:
        foundpy = True

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents.input = -1
process.OtherThing = cms.EDProducer("OtherThingProducer")
process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
    setRunNumber = cms.untracked.uint32(621),
    fileNames = cms.untracked.vstring(argv[0]),
    enforceGUIDInFileName = cms.untracked.bool(True)
)

process.p = cms.Path(process.OtherThing*process.Analysis)


