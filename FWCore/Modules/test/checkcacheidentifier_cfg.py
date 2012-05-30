import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(4)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1)
)

process.MessageLogger = cms.Service("MessageLogger")

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.checker = cms.EDAnalyzer("EventSetupCacheIdentifierChecker",
                              GadgetRcd = cms.untracked.vuint32(2,2,2,2,2,2,2,2,2,3,3,3)
)

process.p = cms.Path(process.checker)
