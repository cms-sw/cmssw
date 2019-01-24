import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.string('DetectorDescription/DDCMS/data/cms-mf-geometry.xml')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile")

process.p = cms.Path(process.testDump)
