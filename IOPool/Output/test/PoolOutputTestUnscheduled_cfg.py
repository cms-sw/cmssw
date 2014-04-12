import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.options.allowUnscheduled = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.intProducer1 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(7)
)

process.intProducer2 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(11)
)

process.aliasForInt1 = cms.EDAlias(
    intProducer1 = cms.VPSet(
        cms.PSet(type = cms.string('edmtestIntProduct'))
    )
)

process.aliasForInt2 = cms.EDAlias(
    intProducer2 = cms.VPSet(
        cms.PSet(type = cms.string('edmtestIntProduct'))
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputTestUnscheduled.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducer1_*_*',
        'drop *_aliasForInt1_*_*',
        'drop *_intProducer2_*_*'
    )
)

process.getInt = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(
        cms.InputTag("aliasForInt1"),
    ),
  expectedSum = cms.untracked.int32(140)
)

process.source = cms.Source("EmptySource")

process.path1 = cms.Path(process.getInt)

process.ep = cms.EndPath(process.output)


