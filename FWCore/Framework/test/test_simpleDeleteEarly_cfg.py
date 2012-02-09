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
                                expectedValues = cms.untracked.vuint32(2,4,6))

process.p = cms.Path(process.maker+process.reader+process.tester)
