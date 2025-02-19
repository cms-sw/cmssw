#No edmtest::IntProduct's added to this output file
import FWCore.ParameterSet.Config as cms
process = cms.Process("PROD")

process.source = cms.Source("EmptySource",
                                firstLuminosityBlock = cms.untracked.uint32(2),
                                firstEvent = cms.untracked.uint32(20)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.c = cms.EDProducer("ThingProducer")

process.d = cms.EDProducer("OtherThingProducer",
                           thingLabel=cms.untracked.string("c"))

process.o = cms.OutputModule("PoolOutputModule",
                             outputCommands = cms.untracked.vstring("drop *",
                                                                    "keep edmtestThings_*_*_*",
                                                                    "keep edmtestOtherThings_*_*_*"),
                             fileName = cms.untracked.string("ref_merge_prod2.root")
                             )

process.p = cms.Path(process.c*process.d)

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
                                other = cms.untracked.InputTag("d","testUserTag"))

process.out = cms.EndPath(process.o+process.tester)


                             

                            
