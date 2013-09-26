# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTANALYSIS")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")

process.source = cms.Source("PoolSource",
    setRunNumber = cms.untracked.uint32(621),
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.p = cms.Path(process.Analysis)

#Also test creating a Ref from the alias branch
process.OtherThing2 = cms.EDProducer("OtherThingProducer",
                                    thingTag = cms.InputTag("AltThing")
                                    )

process.Analysis2 = cms.EDAnalyzer("OtherThingAnalyzer",
                                   other=cms.untracked.InputTag("OtherThing2","testUserTag"))

process.p2 = cms.Path(process.OtherThing2+process.Analysis2)
