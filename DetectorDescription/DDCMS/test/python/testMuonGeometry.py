import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            label = cms.string('CMS')
                                            )
process.DDDetectorESProducer2 = cms.ESSource("DDDetectorESProducer",
                                             confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-mf-geometry.xml'),
                                             label = cms.string('MagneticField')
                                             )
process.DDDetectorESProducer3 = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            label = cms.string('')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    label = cms.string('CMS'))

process.test = cms.EDAnalyzer("DDCMSDetector",
                              fromDataLabel = cms.untracked.string('CMS')
                              )
process.DDVectorRegistryESProducer2 = cms.ESProducer("DDVectorRegistryESProducer",
                                                    label = cms.string('MagneticField'))
process.test2 = cms.EDAnalyzer("DDCMSDetector",
                              fromDataLabel = cms.untracked.string('MagneticField')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors",
                                     fromDataLabel = cms.untracked.string('CMS')
                                     )
process.testDump = cms.EDAnalyzer("DDTestDumpFile")
process.testGeoIter = cms.EDAnalyzer("DDTestDumpGeometry")

process.p = cms.Path(
    process.test
    +process.test2
    +process.testVectors
    ##+process.testDump
    +process.testGeoIter)
