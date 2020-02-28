# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms
from sys import argv

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int(argv[2]))
)

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
    setRunNumberForEachLumi = cms.untracked.vuint32(111,222,333,444,555),
    fileNames = cms.untracked.vstring('file:RunPerLumiTest.root')
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('OutputRunPerLumiTest.root')
)

process.p = cms.Path(process.OtherThing*process.Analysis)

process.e = cms.EndPath(process.output)

