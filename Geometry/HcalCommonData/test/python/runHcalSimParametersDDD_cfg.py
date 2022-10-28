import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process("HcalParametersTest",Run3_DDD)

process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()

process.hpa = cms.EDAnalyzer("HcalSimParametersAnalyzer")

process.p1 = cms.Path(process.hpa)
