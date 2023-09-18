# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms
from sys import argv

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents.input = int(argv[1])

runToLumi = [111,222,333,444,555]

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
                            setRunNumberForEachLumi = cms.untracked.vuint32(*runToLumi),
                            fileNames = cms.untracked.vstring('file:RunPerLumiTest.root')
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('OutputRunPerLumiTest.root')
)

numberOfEventsInLumi = 5
ids = cms.VEventID()
for l,r in enumerate(runToLumi):
    for e in range(numberOfEventsInLumi):
        ids.append(cms.EventID(r, l+1,l*5+e+1))

process.check = cms.EDAnalyzer("EventIDChecker", eventSequence = cms.untracked(ids))


process.p = cms.Path(process.OtherThing*process.Analysis)

process.e = cms.EndPath(process.check+process.output)
