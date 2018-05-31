import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                                                         'Geometry/CMSCommonData/data/rotations.xml',
                                                         'Geometry/TrackerCommonData/data/pixbarmaterial.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarladder.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarladderfull.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarladderhalf.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarlayer.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarlayer0.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarlayer1.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbarlayer2.xml', 
                                                         'Geometry/TrackerCommonData/data/pixbar.xml', 
                                                         'Geometry/TrackerCommonData/data/trackerpixbar.xml', 
                                                         'Geometry/TrackerCommonData/data/tracker.xml',
                                                         'Geometry/TrackerCommonData/data/trackermaterial.xml',
                                                         'Geometry/TrackerCommonData/data/pixfwdMaterials.xml',
                                                         'Geometry/CMSCommonData/data/cmsMother.xml',
                                                         'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
                                                         'Geometry/CMSCommonData/data/cms.xml'),
                              confGeomXMLFiles = cms.string('DetectorDescription/DDCMS/data/cms-test-shapes.xml')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors")
process.testDump = cms.EDAnalyzer("DDTestDumpFile")

process.p = cms.Path(process.test+process.testVectors+process.testDump)
