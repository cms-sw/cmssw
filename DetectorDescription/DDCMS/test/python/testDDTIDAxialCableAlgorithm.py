import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath(
        'DetectorDescription/DDCMS/data/cms-test-tid-axial-cable-algorithm.xml'
    ),
    appendToDataLabel = cms.string('TestTIDAxialCable')
)

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
    DDDetector = cms.ESInputTag('TestTIDAxialCable')
)

process.p = cms.Path(process.testDump)
