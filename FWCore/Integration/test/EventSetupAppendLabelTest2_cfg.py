import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3)
)

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer",
    doodadLabel = cms.string('Two')
)

process.DoodadESSource = cms.ESSource("DoodadESSource",
    appendToDataLabel = cms.string('Two')
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::WhatsIt', 
                           'edmtest::Doodad/Two')
    )),
    verbose = cms.untracked.bool(True)
)

process.printIt = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.get)
process.ep = cms.EndPath(process.printIt)
