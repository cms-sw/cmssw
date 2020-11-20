import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHcalTBHOGeomTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HcalAlgo/data/cms-test-ddhcalTBZpos-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDHCalTBHO')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('tbHODD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDHCalTBHO')
                                  )

process.p = cms.Path(process.testDump)
