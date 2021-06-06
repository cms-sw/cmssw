import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHGCalHEsilTest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HGCalCommonData/data/dd4hep/cms-test-ddhgcalHEsil-algorithm.xml'),
                                            appendToDataLabel = cms.string('DDHGCalHEsil')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('hgcalHEsilDD4Hep.root'),
                                  DDDetector = cms.ESInputTag('','DDHGCalHEsil')
                                  )

process.p = cms.Path(process.testDump)
