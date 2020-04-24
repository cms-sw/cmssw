import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Ideal geometry producer
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# testing ideal geometry from Fake...
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

##    // ...or testing geometry from DB/sqlite...
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
#
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#             CondDBSetup,
#             #connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
#             connect = cms.string('sqlite:Alignments.db'),
#             toGet = cms.VPSet(cms.PSet(
#    record = cms.string('TrackerAlignmentRcd'),
#    tag = cms.string('TrackerCRAFTScenario22X_mc') # TrackerIdealGeometry210_mc Tracker10pbScenario210_mc
#    ), 
#                               cms.PSet(
#    record = cms.string('TrackerAlignmentErrorExtendedRcd'),
#    tag = cms.string('TrackerCRAFTScenarioErrors22X_mc') # TrackerIdealGeometryErrors210_mc Tracker10pbScenarioErrors210_mc
#    ))
#             )

## ... or testing any of the scenarios from cff
#process.load("Alignment.TrackerAlignment.MisalignedTracker_cfi")
#import Alignment.TrackerAlignment.Scenarios_cff as _Scenarios
##process.MisalignedTracker.scenario = _Scenarios.TrackerSurveyLASOnlyScenario
##process.MisalignedTracker.scenario = _Scenarios.TrackerSurveyLASCosmicsScenario
#process.MisalignedTracker.scenario = _Scenarios.TrackerCRAFTScenario
##process.MisalignedTracker.scenario = _Scenarios.Tracker1000pbScenario

# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("TestTrackerHierarchy",
    dumpAlignments = cms.untracked.bool(False) # (True)

)

process.p1 = cms.Path(process.prod)


