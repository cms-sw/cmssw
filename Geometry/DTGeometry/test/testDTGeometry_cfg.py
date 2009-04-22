import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.fromDDD = cms.bool(True)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_31X::All' 
#process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.load("CondCore.DBCommon.CondDBCommon_cfi")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("DTGeometryAnalyzer")

process.p1 = cms.Path(process.prod)
#process.e = cms.EndPath(process.out)


