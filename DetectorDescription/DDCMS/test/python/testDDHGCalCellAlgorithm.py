import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHGCalCellTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-test-ddhgcalcell-algorithm.xml'),
                                            appendToDataLabel = cms.string('TestDDHGCalCell')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','TestDDHGCalCell')
                                  )

process.p = cms.Path(process.testDump)
