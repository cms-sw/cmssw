import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))

process.options = cms.untracked.PSet(
        canDeleteEarly = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))


process.maker = cms.EDProducer("DeleteEarlyProducer")

process.reader = cms.EDAnalyzer("DeleteEarlyReader",
                                tag = cms.untracked.InputTag("maker"),
                                mightGet = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))

process.tester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(1,3,5))

process.p = cms.Path(process.maker+process.reader+process.tester)

process2 = cms.Process("SUB")
process2.options = cms.untracked.PSet(
                                     canDeleteEarly = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))
process2.reader = cms.EDAnalyzer("DeleteEarlyReader",
                                tag = cms.untracked.InputTag("maker"),
                                mightGet = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))
#NOTE: even though it was marked for early delete, the original process still has a shared pointer to the object
# and therefore it doesn't get deleted
process2.tester2 = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(1,3,5))
process2.p = cms.Path(process2.reader+process2.tester2)

process.add_(cms.SubProcess(process2))
