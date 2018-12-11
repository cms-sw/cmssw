import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
                                                         'Geometry/CMSCommonData/data/cms.xml', 
                                                         'DetectorDescription/DDCMS/data/cmsMagneticField.xml', 
                                                         'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_1.xml',
                                                         'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_2.xml',
                                                         'Geometry/CMSCommonData/data/materials.xml'),
                              confGeomXMLFiles = cms.string('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors")
process.testDump = cms.EDAnalyzer("DDTestDumpFile")
##process.testGeoIter = cms.EDAnalyzer("DDTestDumpGeometry")

process.p = cms.Path(process.test+process.testVectors+process.testDump) ##+process.testGeoIter)
