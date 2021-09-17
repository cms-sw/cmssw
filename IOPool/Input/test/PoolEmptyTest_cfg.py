# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITEEMPTY")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.Thing = cms.EDProducer("ThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PoolEmptyTest.root')
)

# Select lumis that we know are not in the input file so
# that the output file is empty (contains no Runs, Lumis,
# or Events, but it does have metadata)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:PoolInputTest.root'),
                            lumisToProcess = cms.untracked.VLuminosityBlockRange('1:1')
)

process.p = cms.Path(process.Thing)
process.ep = cms.EndPath(process.output)

