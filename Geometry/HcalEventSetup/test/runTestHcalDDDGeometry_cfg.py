import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("Geometry.CMSCommonData.hcalOnlyGeometryXML_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('HCalGeom'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load("Geometry.HcalEventSetup.HcalDDDGeometry_cfi")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

process.prod = cms.EDAnalyzer("HcalDDDGeometryAnalyzer")

process.p1 = cms.Path(process.prod)
