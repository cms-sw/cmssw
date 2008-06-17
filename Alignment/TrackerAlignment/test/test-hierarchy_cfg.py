import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Ideal geometry producer
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# testing ideal geometry from Fake
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

#    // testing geometry from DB/sqlite
#   include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
#   include "Alignment/CommonAlignmentProducer/data/GlobalPosition_Frontier.cff"
#   replace TrackerDigiGeometryESModule.applyAlignment = true
#
#    include "CondCore/DBCommon/data/CondDBSetup.cfi"
#    es_source myTrackerAlignmentSource = PoolDBESSource {
#     using CondDBSetup
#     #string connect = "sqlite_file:TrackerGeometry_fix.db"    
#     string connect = "frontier://FrontierProd/CMS_COND_20X_ALIGNMENT"
#     VPSet toGet = {
#           { string record = "TrackerAlignmentRcd" string tag = "Tracker10pbScenario200_mc" },
#           { string record = "TrackerAlignmentErrorRcd" string tag = "Tracker10pbScenarioErrors200_mc" }
#           { string record = "TrackerAlignmentRcd" string tag = "TrackerIdealGeometry200_v2" },
#           { string record = "TrackerAlignmentErrorRcd" string tag = "TrackerIdealGeometryErrors200_v2" }
#     }
#   }
# testing any of the scenarios from cff
#    include "Alignment/TrackerAlignment/data/ExampleScenario.cff"
#    include "Alignment/TrackerAlignment/data/NoMovementsScenario.cff"
#    include "Alignment/TrackerAlignment/data/SurveyLASOnlyScenario.cff"
#    include "Alignment/TrackerAlignment/data/SurveyLASCosmicsScenario.cff"
#    include "Alignment/TrackerAlignment/data/Tracker1000pbScenario.cff"
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDFilter("TestTrackerHierarchy",
    dumpAlignments = cms.untracked.bool(False) ## true 

)

process.p1 = cms.Path(process.prod)
process.MessageLogger.cout = cms.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


