import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process('Dump',Run3_dd4hep)

process.load('Configuration.Geometry.GeometryDD4hepExtended2021_cff')
process.load('Geometry.MuonNumbering.muonOffsetESProducer_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

if hasattr(process,'MessageLogger'):
    process.MessageLogger.MuonGeom=dict()

process.hpa = cms.EDAnalyzer("MuonOffsetAnalyzer")
process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.hpa)
