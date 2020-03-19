# Configuration file to run stubs/CSCGeometryAsLayers
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
    errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        extension = cms.untracked.string('.out')
    ),
    # No constraint on log content...equivalent to threshold INFO
    # 0 means none, -1 means all (?)
    log = cms.untracked.PSet(
        extension = cms.untracked.string('.out')
    ),
    debug = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        extension = cms.untracked.string('.out'),
        CSC = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CSCNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
        CSCGeometryBuilderFromDDD = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        RadialStripTopology = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    # For LogDebug/LogTrace output...
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('CSC', 
        'CSCNumbering', 
        'CSCGeometryBuilderFromDDD', 
        'RadialStripTopology'),
    destinations = cms.untracked.vstring('log', 
        'errors', 
        'debug')
)

process.producer = cms.EDAnalyzer("CSCGeometryAsLayers")

process.p1 = cms.Path(process.producer)
process.CSCGeometryESModule.debugV = True

