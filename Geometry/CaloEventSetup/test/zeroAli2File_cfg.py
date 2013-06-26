import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:zeroEcalAlignments.db'),
    toPut = cms.VPSet(
      cms.PSet(
        record = cms.string('EBAlignmentRcd'),
        tag = cms.string('EB')
      ),
      cms.PSet(
        record = cms.string('EEAlignmentRcd'),
        tag = cms.string('EE')
      ),
      cms.PSet(
        record = cms.string('ESAlignmentRcd'),
        tag = cms.string('ES')
      )
    )
)

process.load('Geometry.CaloEventSetup.FakeCaloAlignments_cff')

process.CaloAlignmentRcdWrite = cms.EDAnalyzer("CaloAlignmentRcdWrite")

process.p = cms.Path(process.CaloAlignmentRcdWrite)
