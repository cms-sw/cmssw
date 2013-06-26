import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.load('Geometry.CaloEventSetup.FakeCaloAlignments_cff')

process.CaloAlignmentRcdRead = cms.EDAnalyzer("CaloAlignmentRcdRead")

process.p = cms.Path(process.CaloAlignmentRcdRead)
