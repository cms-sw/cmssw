import FWCore.ParameterSet.Config as cms

process = cms.Process("Analyze")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:ref_alias_drop_alias.root"))

process.otherThing3 = cms.EDProducer("OtherThingProducer",
                                     thingTag=cms.InputTag("thing"))

process.comparerA = cms.EDAnalyzer("OtherThingRefComparer", 
                                  first = cms.untracked.InputTag("otherThing1:testUserTag"),
                                  second = cms.untracked.InputTag("otherThing2:testUserTag")
                                  )

process.comparerB = cms.EDAnalyzer("OtherThingRefComparer", 
                                  first = cms.untracked.InputTag("otherThing1:testUserTag"),
                                  second = cms.untracked.InputTag("otherThing3:testUserTag")
                                 )

process.comparerC = cms.EDAnalyzer("OtherThingRefComparer", 
                                   first = cms.untracked.InputTag("otherThing2:testUserTag"),
                                   second = cms.untracked.InputTag("otherThing3:testUserTag")
                                  )

process.p = cms.Path(process.otherThing3+process.comparerA+process.comparerB+process.comparerC)
