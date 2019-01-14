import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-mf-geometry.xml'),
                                            label = cms.string('MagneticField')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  fromDataLabel = cms.untracked.string('MagneticField')
                                  )

process.p = cms.Path(process.testDump)
