import FWCore.ParameterSet.Config as cms

process = cms.Process("DTT0Analyzer")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.GlobalTag.globaltag = ''

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.DTGeometryESModule.fromDDD = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dtT0Analyzer = cms.EDAnalyzer("DTT0Analyzer",
    rootFileName = cms.untracked.string("") 
)

process.p = cms.Path(process.dtT0Analyzer)
