import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.options = cms.untracked.PSet(
        canDeleteEarly = cms.untracked.vstring("edmtestDeleteEarly_maker__TEST"))
#process.options.holdsReferencesToDeleteEarly = [cms.PSet(product=cms.string("edmtestDeleteEarlyedmRefProd_ref__TEST"), references=cms.vstring("edmtestDeleteEarly_maker__TEST"))]

process.maker = cms.EDProducer("DeleteEarlyProducer")

process.ref = cms.EDProducer("DeleteEarlyRefProdProducer", get = cms.InputTag("maker"))

process.reader = cms.EDAnalyzer("DeleteEarlyRefProdReader",
                                tag = cms.untracked.InputTag("ref"))

process.testerBefore = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(1,3,5))

process.testerAfter = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(2,4,6))

process.p = cms.Path(process.maker+process.ref+cms.wait(process.testerBefore)+process.reader+cms.wait(process.testerAfter))

