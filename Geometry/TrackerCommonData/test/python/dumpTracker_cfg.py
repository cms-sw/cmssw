import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.load("Geometry.TrackerCommonData.cmsExtendedGeometry2017XML_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

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

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))


process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string('track2007.root')
)

process.p = cms.Path(process.dump)
