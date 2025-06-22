# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedestals_v1')
    ), 
        cms.PSet(
            record = cms.string('SiStripNoisesRcd'),
            tag = cms.string('SiStripNoises_v1')
        ), 
        cms.PSet(
            record = cms.string('SiStripFedCablingRcd'),
            tag = cms.string('SiStripFedCabling_v1')
        )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_STRIP'), ##cms_conditions_data/CMS_COND_STRIP"

)

process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(5),
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        data = cms.vstring('SiStripPedNoise_MTCC_v1_p')
    ), 
        cms.PSet(
            record = cms.string('SiStripNoisesRcd'),
            data = cms.vstring('SiStripPedNoise_MTCC_v1_n')
        ), 
        cms.PSet(
            record = cms.string('SiStripFedCablingRcd'),
            data = cms.vstring('SiStripCabling_MTCC_v1')
        )),
    verbose = cms.untracked.bool(True)
)

process.printer = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.get)
process.ep = cms.EndPath(process.printer)

