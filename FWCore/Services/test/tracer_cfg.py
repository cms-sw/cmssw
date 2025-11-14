# Configuration file for Tracer service

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.Tracer = cms.Service("Tracer",
                         useMessageLogger=cms.untracked.bool(False),
                         fileName=cms.untracked.string("tracer.log"))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

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
    ))
)

process.print1 = cms.OutputModule("AsciiOutputModule")

process.print2 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.print1*process.print2+process.get)


