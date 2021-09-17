import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHGCalWaferAlgoTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HGCalCommonData/data/dd4hep/cms-test-ddhgcalWaferAlgo-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDHGCalWaferAlgo')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('hgcalWaferAlgoDD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDHGCalWaferAlgo')
                                  )

process.p = cms.Path(process.testDump)
