import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHGCalWafer8Test")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-test-ddhgcalwafer8-algorithm.xml'),
                                            appendToDataLabel = cms.string('TestDDHGCalWafer8')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','TestDDHGCalWafer8')
                                  )

process.p = cms.Path(process.testDump)
