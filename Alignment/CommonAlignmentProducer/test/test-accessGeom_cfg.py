import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("FWCore.MessageService.MessageLogger_cfi")

#    service = Tracer {}
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.DTGeometryBuilder.dtGeometry_cfi")

process.load("Geometry.CSCGeometryBuilder.cscGeometry_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#    include "Configuration/StandardSequences/data/FakeConditions.cff"
#    replace TrackerDigiGeometryESModule.applyAlignment = true
#    replace DTGeometryESModule.applyAlignment = true
#    replace CSCGeometryESModule.applyAlignment = true
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")

process.load("Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff")

process.load("Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.demo = cms.EDFilter("TestAccessGeom",
    # spaces are removed internally - but are needed since framework seems to ignore empty strings...
    TrackerGeomLabels = cms.vstring('idealForDigi', 
        ' '),
    DTGeomLabels = cms.vstring('idealForDigi', 
        ' '),
    CSCGeomLabels = cms.vstring('idealForDigi', 
        ' ')
)

process.p = cms.Path(process.demo)
process.MessageLogger.cerr.INFO.limit = 1000000
process.MessageLogger.cerr.noTimeStamps = True


