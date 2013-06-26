import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
#process.load("DetectorDescription.OfflineDBLoader.test.cmsIdealGeometryForWrite_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.XMLIdealGeometryESSource.geomXMLFiles = cms.vstring("runB/fred.xml")
#process.XMLIdealGeometryESSource.userControlledNamespace = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.myprint = cms.OutputModule("AsciiOutputModule")

#MuonSensitiveDetector  ReadOutName MuonCSCHits
process.prod = cms.EDAnalyzer("TestSpecParAnalyzer"
                                  , specName = cms.string("ReadOutName")
                                  , specStrValue = cms.untracked.string("MuonCSCHits")
                                  #, specDblValue = cms.untracked.double(0.0)
                              )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)
