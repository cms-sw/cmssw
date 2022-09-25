import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))

process.options = cms.untracked.PSet(
        canDeleteEarly = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"),
        modulesToIgnoreForDeleteEarly = cms.untracked.vstring("consumer2")
)


process.maker = cms.EDProducer("DeleteEarlyProducer")

process.reader = cms.EDAnalyzer("DeleteEarlyReader",
                                tag = cms.untracked.InputTag("maker"))

process.tester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
    expectedValues = cms.untracked.vuint32(2,4,6)
)

# the following consumes the DeleteEarly but does not read it
# it won't fail as such, but triggers a prefetch request to maker
process.consumer2 = cms.EDAnalyzer("DeleteEarlyConsumer",
                                    tag = cms.untracked.InputTag("maker"))

process.t = cms.Task(process.maker)
process.p = cms.Path(cms.wait(process.reader)+process.tester+process.consumer2)
process.p.associate(process.t)
