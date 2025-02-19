import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
process.load('Configuration.Geometry.GeometryExtended_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')

process.TrackerGeometricDetExtraESModule = cms.ESProducer( "TrackerGeometricDetExtraESModule",
                                                           fromDDD = cms.bool( True )
                                                           )

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.out = cms.OutputModule("AsciiOutputModule")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('*')
process.MessageLogger.categories.append('TrackerGeometryBuilder')
process.MessageLogger.categories.append('TrackerNumberingBuilder')
process.MessageLogger.categories.append('ModuleInfo')
process.MessageLogger.categories.append('CmsTrackerDebugNavigator')
process.MessageLogger.cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
            default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
            TrackerNumberingBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            TrackerGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            ModuleInfo = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            CmsTrackerDebugNavigator =cms.untracked.PSet( limit = cms.untracked.int32(-1) )
            )

process.prod = cms.EDAnalyzer("ModuleInfo",
    fromDDD = cms.bool(True),
    printDDD = cms.untracked.bool(False)
)

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)


