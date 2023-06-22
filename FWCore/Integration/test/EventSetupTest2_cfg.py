import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3)
)

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.emptyESSourceK = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordK"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8),
    iovIsRunNotTime = cms.bool(True)
)

process.testESProductResolverProviderJ = cms.ESProducer("ESTestESProductResolverProviderJ",
    expectedCacheIds = cms.untracked.vuint32(2, 3, 4, 5, 6, 7, 8)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('GadgetRcd'),
            data = cms.vstring('edmtest::WhatsIt',
                               'edmtest::Doodad')
        )
    ),
    verbose = cms.untracked.bool(True)
)

process.esAnalyzerJ = cms.EDAnalyzer("ESTestAnalyzerJ")

process.printIt = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.get * process.esAnalyzerJ)
process.ep = cms.EndPath(process.printIt)
