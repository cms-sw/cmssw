import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-tracker.xml'),
                                            label = cms.string('CMS')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    label = cms.string('CMS'))

process.test = cms.EDAnalyzer("DDCMSDetector",
                              fromDataLabel = cms.untracked.string('CMS')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors",
                                     fromDataLabel = cms.untracked.string('CMS')
                                     )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  fromDataLabel = cms.untracked.string('CMS')
                                  )

process.p = cms.Path(process.test+process.testVectors+process.testDump)
