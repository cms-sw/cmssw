import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTowerGeometryTest")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.ctga = cms.EDAnalyzer("CaloTowerGeometryAnalyzer",
                              Epsilon = cms.double(0.004),
                              FileName = cms.string("CaloTower.cells"))


process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.ctga)
