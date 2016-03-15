import FWCore.ParameterSet.Config as cms

process = cms.Process("BasicFrameworkTest")

#===============================================================================
# Logging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
  #statistics   = cms.untracked.vstring('trackerGeometryAnalyzer'),
  destinations = cms.untracked.vstring('basicFrameworkTest'),
  categories   = cms.untracked.vstring(
    'BasicFrameworkTest',
  ),

  basicFrameworkTest = cms.untracked.PSet(
    INFO    = cms.untracked.PSet(limit = cms.untracked.int32( 0)),
    DEBUG   = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
    WARNING = cms.untracked.PSet(limit = cms.untracked.int32( 0)),
    ERROR   = cms.untracked.PSet(limit = cms.untracked.int32(-1)),

    threshold = cms.untracked.string('DEBUG'),
    
    #AlignableMap            = cms.untracked.PSet(limit = cms.untracked.int32(-1))
  )
)

#===============================================================================
# Geometry
process.load("Configuration.Geometry.GeometryIdeal_cff") # uses ESProducer XMLIdealGeometryESSource
# current geometry:
#process.load("Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi")
# phase-I upgrade geometry:
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2016XML_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi")
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
process.BasicFrameworkTest = cms.EDAnalyzer(
  "BasicFrameworkTest",
  TopFolderName = cms.string("PixelPhase1")
)
process.p1 = cms.Path(
  process.BasicFrameworkTest
)
