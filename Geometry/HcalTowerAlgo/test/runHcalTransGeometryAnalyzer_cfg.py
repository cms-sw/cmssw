import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalGeometryTest")

process.load("Configuration.Geometry.GeometryExtended2023D3_cff")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP" ,
                                                UseOldLoader   = cms.bool(False),
                                                appendToDataLabel = cms.string("_master")
                                                )
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hga = cms.EDAnalyzer("HcalGeometryAnalyzer",
                             UseOldLoader   = cms.bool(False),
                             GeometryFromDB = cms.bool(False))

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
