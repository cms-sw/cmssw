import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-test-shapes.xml'),
                                            label = cms.string('TestShapes')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    label = cms.string('TestShapes')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              fromDataLabel = cms.untracked.string('TestShapes')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors",
                                     fromDataLabel = cms.untracked.string('TestShapes')
                                     )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  fromDataLabel = cms.untracked.string('TestShapes')
                                  )

process.p = cms.Path(process.test+process.testVectors+process.testDump)
