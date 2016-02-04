import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3)
)

process.MessageLogger = cms.Service("MessageLogger")

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.demo = cms.EDAnalyzer("EventSetupRecordDataGetter",
                              toGet = cms.VPSet(), #empty means get them all
                              verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.demo)
