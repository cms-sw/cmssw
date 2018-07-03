import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalGeometryTest")

#process.load("Geometry.HcalCommonData.testGeometry17bXML_cfi")
process.load("Geometry.HcalCommonData.testPhase2GeometryFineXML_cfi")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hga = cms.EDAnalyzer("HcalGeometryTester",
                             UseOldLoader      = cms.bool(False))

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)

process.hcalTopologyIdeal.MergePosition = False
