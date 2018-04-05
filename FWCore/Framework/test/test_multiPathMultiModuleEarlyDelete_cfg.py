import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(6))

process.options = cms.untracked.PSet(
        canDeleteEarly = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))


process.maker = cms.EDProducer("DeleteEarlyProducer")

process.reader1 = cms.EDAnalyzer("DeleteEarlyReader",
                                tag = cms.untracked.InputTag("maker"),
                                mightGet = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))

process.reader2 = cms.EDAnalyzer("DeleteEarlyReader",
                                 tag = cms.untracked.InputTag("maker"),
                                 mightGet = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))

process.tester1 = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(3,7,11))
process.tester2 = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(6,12))
process.tester3 = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(2,4,6,8,10,12))

process.p2PreTester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                     expectedValues = cms.untracked.vuint32(1,3,5,7,9,11))
process.f2 = cms.EDFilter("TestFilterModule",
onlyOne = cms.untracked.bool(True),
acceptValue = cms.untracked.int32(2)
)

process.f3 = cms.EDFilter("TestFilterModule",
onlyOne = cms.untracked.bool(True),
acceptValue = cms.untracked.int32(3)
)

process.p1Done = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.waitTillP1Done = cms.EDAnalyzer("IntConsumingAnalyzer",
                                        getFromModule = cms.untracked.InputTag("p1Done"))

process.p2Done = cms.EDProducer("IntProducer", ivalue = cms.int32(2))
process.waitTillP2Done = cms.EDAnalyzer("IntConsumingAnalyzer",
                                        getFromModule = cms.untracked.InputTag("p2Done"))


process.p1 = cms.Path(process.maker+process.f2+process.reader1+process.tester1+process.p1Done)
process.p2 = cms.Path(process.waitTillP1Done+process.maker+process.p2PreTester+process.f3+process.reader2+process.tester2+process.p2Done)
process.p3 = cms.Path(process.waitTillP2Done+process.tester3)
