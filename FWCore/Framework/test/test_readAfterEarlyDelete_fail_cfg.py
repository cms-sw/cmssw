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

#the following wants the DeleteEarly but does not say it needs it so will fail
process.readerFail = cms.EDAnalyzer("DeleteEarlyReader",
                                    tag = cms.untracked.InputTag("maker"))


process.p = cms.Path(process.maker+cms.wait(process.reader)+process.readerFail)
process.add_(cms.Service("Tracer"))
