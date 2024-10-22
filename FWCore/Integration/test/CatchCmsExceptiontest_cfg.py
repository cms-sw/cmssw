import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.MessageLogger = cms.Service("MessageLogger")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3)
)

#stuck something into the EventSetup
process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")
#es_source = DoodadESSource {}

process.demo = cms.EDAnalyzer("WhatsItAnalyzer",
    expectedValues = cms.untracked.vint32(0)
)

process.bad = cms.ESSource("EmptyESSource",
    recordName = cms.string('GadgetRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.p = cms.Path(process.demo)
