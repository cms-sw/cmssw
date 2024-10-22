import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("Geometry.CMSCommonData.hcalOnlyGeometryXML_cfi")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

#process.load("Configuration.StandardSequences.MagneticField_cff")

process.load('FWCore.MessageService.MessageLogger_cfi')
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HCalGeom=dict()

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

process.prod = cms.EDAnalyzer("HcalDDDGeometryAnalyzer")

process.p1 = cms.Path(process.prod)
