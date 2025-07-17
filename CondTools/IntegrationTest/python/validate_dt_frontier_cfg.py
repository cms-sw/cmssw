# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('MTCC_t0')
    ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('MTCC_tTrig')
        ), 
        cms.PSet(
            record = cms.string('DTReadOutMappingRcd'),
            tag = cms.string('MTCC_map')
        )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_DT'), ##cms_conditions_data/CMS_COND_DT" 

)

process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(5),
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        data = cms.vstring('DTT0')
    ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            data = cms.vstring('DTTtrig')
        ), 
        cms.PSet(
            record = cms.string('DTReadOutMappingRcd'),
            data = cms.vstring('DTReadOutMapping')
        )),
    verbose = cms.untracked.bool(True)
)

process.printer = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.get)
process.ep = cms.EndPath(process.printer)

