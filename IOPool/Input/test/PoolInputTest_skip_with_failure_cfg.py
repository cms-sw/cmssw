# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.OtherThing = cms.EDProducer("OtherThingProducer")

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
                            skipBadFiles = cms.untracked.bool(True),
                            skipEvents = cms.untracked.uint32(15), #skips all events in first file
                            setRunNumber = cms.untracked.uint32(621),
                            fileNames = cms.untracked.vstring('file:PoolInputTest.root', 
                                                              'file:this_file_doesnt_exist.root')
)

process.p = cms.Path(process.OtherThing*process.Analysis)


