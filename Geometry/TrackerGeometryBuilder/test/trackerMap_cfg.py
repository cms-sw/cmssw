import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")

process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("EmptySource")

process.prod = cms.EDFilter("TrackerMapTool")

process.p1 = cms.Path(process.prod)



