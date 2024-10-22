import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("DTGeometryAnalyzer",
                              tolerance = cms.untracked.double(1.0e-23)
                             )

process.p1 = cms.Path(process.prod)
