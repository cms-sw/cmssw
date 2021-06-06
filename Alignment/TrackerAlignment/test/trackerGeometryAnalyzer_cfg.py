import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerGeometryAnalyzer")

#===============================================================================
# Logging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        trackerGeometryAnalyzer = cms.untracked.PSet(
            AlignableBuildProcess = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            TrackerGeometryAnalyzer = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

#===============================================================================
# Geometry
process.load("Configuration.Geometry.GeometryIdeal_cff") # uses ESProducer XMLIdealGeometryESSource
# current geometry:
process.load("Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi")
# phase-I upgrade geometry:
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2016XML_cfi")
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi")
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2019XML_cfi")
# not working: is PhaseI geometry, but TID values are faulty
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2023XML_cfi")

#===============================================================================
# Event data
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

#===============================================================================
# Condition Data

#===============================================================================
# The actual modules to run
process.TrackerGeometryAnalyzer = cms.EDAnalyzer(
  "TrackerGeometryAnalyzer",

  analyzeAlignables     = cms.bool(True),
  printTrackerStructure = cms.bool(False),
  maxPrintDepth         = cms.int32(10),

  analyzeGeometry = cms.bool(False),
  analyzePXB      = cms.bool(False),
  analyzePXE      = cms.bool(False),
  analyzeTIB      = cms.bool(False),
  analyzeTID      = cms.bool(False),
  analyzeTOB      = cms.bool(False),
  analyzeTEC      = cms.bool(False)
)
process.p1 = cms.Path(
  process.TrackerGeometryAnalyzer
)
