import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-test-ddangular-algorithm.xml'),
                                            label = cms.string('TestAngular')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  fromDataLabel = cms.untracked.string('TestAngular')
                                  )

process.p = cms.Path(process.testDump)
