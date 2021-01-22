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

process.testDataProxyProviderJ = cms.ESProducer("ESTestDataProxyProviderJ",
    expectedCacheIds = cms.untracked.vuint32(2, 3, 4)
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.esAnalyzerL1 = cms.EDAnalyzer("ESTestAnalyzerL",
    src = cms.InputTag("intProducer")
)
process.esAnalyzerL2 = cms.EDAnalyzer("ESTestAnalyzerL",
    src = cms.InputTag("intProducer", "", cms.InputTag.currentProcess())
)

process.p = cms.Path(
    process.intProducer+
    process.esAnalyzerL1+
    process.esAnalyzerL2
)
