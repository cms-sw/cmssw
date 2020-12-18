import FWCore.ParameterSet.Config as cms

process = cms.Process("DDGEMAngularTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.MuonGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MuonCommonData/data/cms-test-ddgemangular-algorithm.xml'),
                                            appendToDataLabel = cms.string('TestDDGEMAngular')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','TestDDGEMAngular')
                                  )

process.p = cms.Path(process.testDump)
