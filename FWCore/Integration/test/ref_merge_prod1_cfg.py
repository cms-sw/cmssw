import FWCore.ParameterSet.Config as cms
process = cms.Process("PROD")

process.source = cms.Source("EmptySource",
                                firstLuminosityBlock = cms.untracked.uint32(1),
                                firstEvent = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.a = cms.EDProducer("IntProducer",
                           ivalue = cms.int32(1))

process.b = cms.EDProducer("IntProducer",
                           ivalue = cms.int32(2))

process.c = cms.EDProducer("ThingProducer")

process.d = cms.EDProducer("OtherThingProducer",
                           thingLabel=cms.untracked.string("c"))

process.o = cms.OutputModule("PoolOutputModule",
                             outputCommands = cms.untracked.vstring("drop *",
                                                                    "keep edmtestThings_*_*_*",
                                                                    "keep edmtestOtherThings_*_*_*"),
                             fileName = cms.untracked.string("ref_merge_prod1.root")
                             )

process.p = cms.Path(process.a+process.b+process.c*process.d)

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
                                other = cms.untracked.InputTag("d","testUserTag"))

process.out = cms.EndPath(process.o+process.tester)


                             

                            
