import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 5

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.PixelGeom=dict()
    process.MessageLogger.TIBGeom=dict()
    process.MessageLogger.TIDGeom=dict()
    process.MessageLogger.TOBGeom=dict()
    process.MessageLogger.TECGeom=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D77.xml'),
                                            appendToDataLabel = cms.string('DDCMS2026D77')
                                            )

process.dump = cms.EDAnalyzer("DDTestDumpFile",
                              outputFileName = cms.untracked.string('cms2026D77.root'),
                              DDDetector = cms.ESInputTag('','DDCMS2026D77')
                              )

process.p = cms.Path(process.dump)
