import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Configuration.Geometry.GeometryExtended2023Dev_cff')
process.load('Geometry.HcalCommonData.hcalParameters_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('HCalGeom'),
    destinations = cms.untracked.vstring('cout')
)

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
