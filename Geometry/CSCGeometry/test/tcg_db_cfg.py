# Configuration file to run stubs/CSCGeometryAnalyser
# I hope this reads geometry from db
# Tim Cox 18.10.2012 for 61X

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCGeometryTest")
process.load("Configuration.Geometry.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.GlobalTag.globaltag = "MC_61_V2::All"
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        CSC = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CSCGeometryBuilderFromDDD = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CSCNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        RadialStripTopology = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.producer = cms.EDAnalyzer("CSCGeometryAnalyzer")

process.p1 = cms.Path(process.producer)
process.CSCGeometryESModule.debugV = True
