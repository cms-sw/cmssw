import FWCore.ParameterSet.Config as cms

process = cms.Process("DDHGCalEETest")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D92.xml'),
                                            appendToDataLabel = cms.string('DDHGCal')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string('CMS2026D92DD4hep.root'),
                                  DDDetector = cms.ESInputTag('','DDHGCal')
                                  )

process.p = cms.Path(process.testDump)
