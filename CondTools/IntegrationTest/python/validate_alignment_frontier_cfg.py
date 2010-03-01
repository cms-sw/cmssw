# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

#					{ string record = "TrackerAlignmentRcd"
#                                         string tag = "TrackerShortTermScenario" },
#					{ string record = "TrackerAlignmentErrorRcd"
#					  string tag = "TrackerShortTermScenarioErrors" },
#					{ string record = "TrackerAlignmentRcd"
#					  string tag = "TrackerLongTermScenario"# },
#	                                 { string record = "TrackerAlignmentErrorRcd"
#					  string tag = "TrackerLongTermScenarioErrors"# }

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    siteLocalConfig = cms.untracked.bool(True),
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('TrackerIdealGeometryErrors')
        )),
    messagelevel = cms.untracked.uint32(0),
    timetype = cms.string('runnumber'),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_ALIGNMENT'), ##cms_conditions_data/CMS_COND_ALIGNMENT"

    authenticationMethod = cms.untracked.uint32(0)
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

