import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("EcalGeometryTest",Run3)

process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.CaloGeometryBuilder.SelectedCalos = ['EcalBarrel', 'EcalEndcap', 'EcalPreshower']

process.demo1 = cms.EDAnalyzer("EcalBarrelCellParameterDump")
process.demo2 = cms.EDAnalyzer("EcalEndcapCellParameterDump")
process.demo3 = cms.EDAnalyzer("EcalPreshowerCellParameterDump")

process.p1 = cms.Path(process.demo1 * process.demo2 * process.demo3)
