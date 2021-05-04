import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-test-tree-navigation.xml'),
                                            appendToDataLabel = cms.string('TestTreeNavigation')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('TestTreeNavigation')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              DDDetector = cms.ESInputTag('','TestTreeNavigation')
                              )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','TestTreeNavigation')
                                  )

process.p = cms.Path(process.test+process.testDump)
