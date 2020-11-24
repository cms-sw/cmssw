# Configuration file to run stubs/CSCGeometryAsChambers
# to dump CSC geometry information - from db
# Tim Cox 18.10.2012 for 61X

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load('Configuration.Geometry.GeometryExtended_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.CommonTopologies.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

process.GlobalTag.globaltag = 'MC_61_V2::All'
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
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            CSC = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            CSCGeometryBuilderFromDDD = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            CSCNumbering = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            RadialStripTopology = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            extension = cms.untracked.string('.out'),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('DEBUG')
        ),
        errors = cms.untracked.PSet(
            extension = cms.untracked.string('.out'),
            threshold = cms.untracked.string('ERROR')
        ),
        log = cms.untracked.PSet(
            extension = cms.untracked.string('.out')
        )
    )
)

process.producer = cms.EDAnalyzer("CSCGeometryAsChambers")

process.p1 = cms.Path(process.producer)
process.CSCGeometryESModule.debugV = True

