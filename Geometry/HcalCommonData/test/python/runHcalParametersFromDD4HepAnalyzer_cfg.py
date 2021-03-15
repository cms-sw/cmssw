import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("HcalParametersTest",Run3_dd4hep)

process.load("Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.Geometry=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
