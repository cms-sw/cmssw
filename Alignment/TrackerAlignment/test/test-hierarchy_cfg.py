import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Ideal geometry producer
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# testing ideal geometry from Fake...
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

#    // ...or testing geometry from DB/sqlite...
# process.load("Geometry/TrackerGeometryBuilder/trackerGeometry_cfi")
# process.load("Alignment/CommonAlignmentProducer/data/GlobalPosition_Frontier_cff")
# process.TrackerDigiGeometryESModule.applyAlignment = True
#
# from CondCore.DBCommon.CondDBSetup_cfi import *
# process.PoolDBESSource =
# cms.ESSource("PoolDBESSource",
#              CondDBSetup,
#              connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
#              toGet = cms.VPSet(cms.PSet(
#     record = cms.string('TrackerAlignmentRcd'),
#     tag = cms.string('TrackerIdealGeometry210_mc') # Tracker10pbScenario210_mc
#     ), 
#                                cms.PSet(
#     record = cms.string('TrackerAlignmentErrorRcd'),
#     tag = cms.string('TrackerIdealGeometryErrors210_mc') # Tracker10pbScenarioErrors210_mc
#     ))
#              )
#
# ... or testing any of the scenarios from cff
# process.load("Alignment.TrackerAlignment.ExampleScenario.cff")
# process.load("Alignment.TrackerAlignment.NoMovementsScenario.cff")
# process.load("Alignment.TrackerAlignment.SurveyLASOnlyScenario.cff")
# process.load("Alignment.TrackerAlignment.SurveyLASCosmicsScenario_cff")
# process.load("Alignment.TrackerAlignment.Tracker1000pbScenario.cff")

# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDFilter("TestTrackerHierarchy",
    dumpAlignments = cms.untracked.bool(False) # (True)

)

process.p1 = cms.Path(process.prod)
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


