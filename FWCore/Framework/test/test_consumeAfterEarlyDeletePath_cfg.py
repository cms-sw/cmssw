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

# the following consumes the DeleteEarly but does not read it nor say it needs it
# it won't fail as such, but triggers a prefetch request to maker
process.consumer2 = cms.EDAnalyzer("DeleteEarlyConsumer",
                                    tag = cms.untracked.InputTag("maker"))

process.p = cms.Path(process.maker+process.reader+process.consumer2)
