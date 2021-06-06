import FWCore.ParameterSet.Config as cms

process = cms.Process("DDAHcalModuleAlgoTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HGCalCommonData/data/dd4hep/cms-test-ddahcalModuleAlgo-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDAHcalModuleAlgo')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('ahcalModuleDD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDAHcalModuleAlgo')
                                  )

process.p = cms.Path(process.testDump)
