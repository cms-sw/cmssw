import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# DT and CSC geometry 
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

# Reading misalignments from DB
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTAlignmentRcd'),
        tag = cms.string('DT1InversepbScenario200_mc')
    ), 
        cms.PSet(
            record = cms.string('DTAlignmentErrorExtendedRcd'),
            tag = cms.string('DT1InversepbScenarioErrors200_mc')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentRcd'),
            tag = cms.string('CSC1InversepbScenario200_mc')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentErrorExtendedRcd'),
            tag = cms.string('CSC1InversepbScenarioErrors200_mc')
        )),
    connect = cms.string('sqlite_file:Alignments.db')
)

process.prod = cms.EDAnalyzer("TestMuonReader")

process.p1 = cms.Path(process.prod)


