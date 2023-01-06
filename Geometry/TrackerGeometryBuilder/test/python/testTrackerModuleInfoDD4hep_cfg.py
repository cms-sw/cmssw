import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("GeometryTest", Run3_dd4hep)
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
                                                      fromDDD = cms.bool( False ),
                                                      fromDD4hep = cms.bool( True )
                                                     )

process.es_prefer_geomdet = cms.ESPrefer("TrackerGeometricDetESModule","")

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.out = cms.OutputModule("AsciiOutputModule")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('*')
process.MessageLogger.cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
            default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
            TrackerNumberingBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            TrackerGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            ModuleInfo = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            )

process.prod = cms.EDAnalyzer("ModuleInfo",
    fromDDD = cms.bool(False),
    printDDD = cms.untracked.bool(False),
    tolerance = cms.untracked.double(1.0e-23)
)

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)


