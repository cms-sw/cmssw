# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerShortTermScenario')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('TrackerShortTermScenarioErrors')
        ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentRcd'),
            tag = cms.string('TrackerLongTermScenario')
        ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('TrackerLongTermScenarioErrors')
        )),
    messagelevel = cms.untracked.uint32(2),
    catalog = cms.untracked.string('relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL'), ##cms_orcoff_int2r/CMS_COND_GENERAL"

    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT'), ##cms_orcoff_int2r/CMS_COND_ALIGNMENT"

    authenticationMethod = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(1),
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        data = cms.vstring('Alignments')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            data = cms.vstring('AlignmentErrors')
        )),
    verbose = cms.untracked.bool(True)
)

process.printer = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.get)
process.ep = cms.EndPath(process.printer)

