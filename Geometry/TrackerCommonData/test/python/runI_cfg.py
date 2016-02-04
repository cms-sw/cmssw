import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("Geometry.TrackerCommonData.testGeometryXML_cfi")

process.VisConfigurationService = cms.Service("VisConfigurationService",
    Views = cms.untracked.vstring('3D Window'),
    ContentProxies = cms.untracked.vstring('Simulation/Core', 
        'Simulation/Geometry', 
        'Simulation/MagField')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.m = cms.EDProducer("GeometryProducer",
    MagneticField = cms.PSet(
        delta = cms.double(1.0)
    ),
    UseMagneticField = cms.bool(False),
    UseSensitiveDetectors = cms.bool(False)
)

process.p1 = cms.Path(process.m)

