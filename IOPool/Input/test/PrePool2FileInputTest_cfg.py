# The following comments couldn't be translated into the new config version:

# Test storing OtherThing as well
# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2ND")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(11)
#)
#process.Thing = cms.EDProducer("ThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_Thing_*_*'),
    fileName = cms.untracked.string('PoolInput2FileTest.root')
)

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:PoolInputOther.root") )


process.p = cms.Path(process.OtherThing)
process.ep = cms.EndPath(process.output)


