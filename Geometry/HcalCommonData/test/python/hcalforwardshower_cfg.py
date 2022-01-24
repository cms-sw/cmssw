import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("Geometry.HcalCommonData.hcalforwardshower_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HCalGeom=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDProducer("GeometryProducer",
    MagneticField = cms.PSet(
        delta = cms.double(1.0)
    ),
    UseMagneticField = cms.bool(False),
    UseSensitiveDetectors = cms.bool(False)
)

process.p1 = cms.Path(process.prod)

