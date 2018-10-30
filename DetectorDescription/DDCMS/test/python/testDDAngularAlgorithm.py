import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              geomXMLFiles = cms.vstring('DetectorDescription/DDCMS/data/cms-test-ddangular-algorithm.xml'),
                              confGeomXMLFiles = cms.string('DetectorDescription/DDCMS/data/cms-test-ddangular-algorithm.xml')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors")
process.testDump = cms.EDAnalyzer("DDTestDumpFile")

process.p = cms.Path(process.test+process.testVectors+process.testDump)
