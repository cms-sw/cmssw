import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.load("Geometry.TrackerCommonData.cmsExtendedGeometry2021FlatMinus05PercentXML_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('PixelGeom')
    process.MessageLogger.categories.append('TIBGeom')
    process.MessageLogger.categories.append('TIDGeom')
    process.MessageLogger.categories.append('TOBGeom')
    process.MessageLogger.categories.append('TECGeom')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))


process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string('track2021FM05.root')
)

process.p = cms.Path(process.dump)
