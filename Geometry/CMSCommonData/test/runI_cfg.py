import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('SimG4CoreGeometry'),
    destinations = cms.untracked.vstring('cout')
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

