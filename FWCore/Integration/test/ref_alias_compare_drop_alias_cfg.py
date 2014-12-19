import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.source = cms.Source("EmptySource")

process.thing = cms.EDProducer("ThingProducer")

process.thingAlias = cms.EDAlias( thing = cms.VPSet(
                                          cms.PSet(type = cms.string('edmtestThings'),
                                                   fromProductInstance = cms.string('*'),
                                                   toProductInstance = cms.string('*'))))

process.otherThing1 = cms.EDProducer("OtherThingProducer",
                                     thingTag=cms.InputTag("thing"))

process.otherThing2 = cms.EDProducer("OtherThingProducer",
                                     thingTag=cms.InputTag("thingAlias"))

process.comparer = cms.EDAnalyzer("OtherThingRefComparer", 
                                  first = cms.untracked.InputTag("otherThing1:testUserTag"),
                                  second = cms.untracked.InputTag("otherThing2:testUserTag")
                                  )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ref_alias_drop_alias.root"),
                               outputCommands = cms.untracked.vstring("keep *",
                               "drop *_thingAlias_*_*")                               
)

process.p = cms.Path(process.thing+process.otherThing1+process.otherThing2+process.comparer)
process.o = cms.EndPath(process.out)