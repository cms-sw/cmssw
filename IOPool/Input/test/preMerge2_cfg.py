import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer",
    useRefs = cms.untracked.bool(True)
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
      'keep *',
      'drop edmtestThings_*__*'
    ),
    fileName = cms.untracked.string('file:big.root')
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)


# foo bar baz
# kNZKUIepMJkuu
# 3jXjthc2Gtzpg
