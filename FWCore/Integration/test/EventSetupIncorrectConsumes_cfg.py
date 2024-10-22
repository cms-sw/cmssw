import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3)
)

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.emptyESSourceK = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordK"),
    firstValid = cms.vuint32(1,2,3),
    iovIsRunNotTime = cms.bool(True)
)

process.testESProductResolverProviderJ = cms.ESProducer("ESTestESProductResolverProviderJ",
    expectedCacheIds = cms.untracked.vuint32(2, 3, 4)
)

process.esAnalyzer = cms.EDAnalyzer("ESTestAnalyzerIncorrectConsumes")

process.p = cms.Path(
    process.esAnalyzer
)
