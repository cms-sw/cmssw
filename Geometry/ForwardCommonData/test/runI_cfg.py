import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('HCalGeom'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

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

