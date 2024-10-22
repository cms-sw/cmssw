import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("EcalGeometryTest",Run3_dd4hep)

process.load('Configuration.Geometry.GeometryDD4hepExtended2021_cff')
process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.demo1 = cms.EDAnalyzer("EcalBarrelCellParameterDump")
process.demo2 = cms.EDAnalyzer("EcalEndcapCellParameterDump")
process.demo3 = cms.EDAnalyzer("EcalPreshowerCellParameterDump")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.demo1 * process.demo2 * process.demo3)
