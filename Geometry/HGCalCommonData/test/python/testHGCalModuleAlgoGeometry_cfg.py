import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHGCalModuleAlgoTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HGCalCommonData/data/dd4hep/cms-test-ddhgcalModuleAlgo-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDHGCalModuleAlgo')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('hgcalModuleAlgoDD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDHGCalModuleAlgo')
                                  )

process.p = cms.Path(process.testDump)
