import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            appendToDataLabel = cms.string('CMS')
                                            )
process.DDDetectorESProducer2 = cms.ESSource("DDDetectorESProducer",
                                             confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-mf-geometry.xml'),
                                             appendToDataLabel = cms.string('MagneticField')
                                             )
process.DDDetectorESProducer3 = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('CMS')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              DDDetector = cms.ESInputTag('','CMS')
                              )

process.DDVectorRegistryESProducer2 = cms.ESProducer("DDVectorRegistryESProducer",
                                                     appendToDataLabel = cms.string('MagneticField')
                                                     )

process.test2 = cms.EDAnalyzer("DDCMSDetector",
                               DDDetector = cms.ESInputTag('','MagneticField')
                               )

process.testVectors = cms.EDAnalyzer("DDTestVectors",
                                     DDDetector = cms.ESInputTag('','CMS')
                                     )

process.testDump = cms.EDAnalyzer("DDTestDumpFile")

process.testGeoIter = cms.EDAnalyzer("DDTestDumpGeometry",
                                     DDDetector = cms.ESInputTag('','CMS')
                                     )

process.p = cms.Path(
    process.test
    +process.test2
    +process.testVectors
    ##+process.testDump
    +process.testGeoIter)
