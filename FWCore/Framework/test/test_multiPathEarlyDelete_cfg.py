import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(6))

process.options = cms.untracked.PSet(
        canDeleteEarly = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))


process.maker = cms.EDProducer("DeleteEarlyProducer")

process.reader = cms.EDAnalyzer("DeleteEarlyReader",
                                tag = cms.untracked.InputTag("maker"),
                                mightGet = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))

process.tester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(2,4,6,8,10,12))

process.p2PreTester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                     expectedValues = cms.untracked.vuint32(1,4,5,8,9,12))
process.f2 = cms.EDFilter("TestFilterModule",
onlyOne = cms.untracked.bool(True),
acceptValue = cms.untracked.int32(2)
)

process.f3 = cms.EDFilter("TestFilterModule",
onlyOne = cms.untracked.bool(True),
acceptValue = cms.untracked.int32(3)
)


process.p1 = cms.Path(process.maker+process.f2+process.reader+process.tester)
process.p2 = cms.Path(process.maker+process.p2PreTester+process.f3+process.reader+process.tester)
process.p3 = cms.Path(process.tester)
