import FWCore.ParameterSet.Config as cms

process = cms.Process("DDFilteredViewTest")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.CMSGeom=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.fva = cms.EDAnalyzer("DDFilteredViewAnalyzer",
                             attribute = cms.string("OnlyForHcalSimNumbering"),
                             value = cms.string("any"))

process.p1 = cms.Path(process.fva)

