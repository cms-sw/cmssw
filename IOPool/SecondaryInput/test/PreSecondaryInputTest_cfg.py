# The following comments couldn't be translated into the new config version:

# Configuration file for PreSecondaryInputTest 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)
process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.EventNumber = cms.EDProducer("EventNumberIntProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('SecondaryInputTest.root')
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.Thing*process.OtherThing*process.EventNumber)
process.ep = cms.EndPath(process.output)


# foo bar baz
# Vv8UH0WiV5JXI
