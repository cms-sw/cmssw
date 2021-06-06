import FWCore.ParameterSet.Config as cms

process = cms.Process("DDForwardShieldTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HcalAlgo/data/cms-test-ddhcalForwardShield-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDHCalFibreBundle')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('forwardShieldDD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDHCalFibreBundle')
                                  )

process.p = cms.Path(process.testDump)
