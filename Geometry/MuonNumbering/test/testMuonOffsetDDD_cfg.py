import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('Dump',Run3)

process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('Geometry.MuonNumbering.muonOffsetESProducer_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('MuonGeom')

process.hpa = cms.EDAnalyzer("MuonOffsetAnalyzer")
process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.hpa)
