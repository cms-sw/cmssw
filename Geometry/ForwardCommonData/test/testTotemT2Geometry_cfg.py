import FWCore.ParameterSet.Config as cms

process = cms.Process("DDTotemT2Test")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.ForwardGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/ForwardCommonData/data/dd4hep/cms-test-ddtotemt2-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDTotemT2')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('totemT2DD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDTotemT2')
                                  )

process.p = cms.Path(process.testDump)
