import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.source = cms.Source("EmptySource")

process.add_( cms.ESProducer("DisplayGeomFromDDD") )
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.dump = cms.EDAnalyzer("DisplayGeom",
                                verbose = cms.untracked.bool(False),
                                level = cms.untracked.int32(11)
                            )

process.p = cms.Path(process.dump)
