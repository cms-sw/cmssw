import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("HcalGeometryTest",Run3_dd4hep)

process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')
process.load("Geometry.HcalTowerAlgo.hcalCellParameterDump_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HCalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.Timing = cms.Service("Timing")

process.hcalParameters.fromDD4hep = cms.bool(True)
process.caloSimulationParameters.fromDD4hep = cms.bool(True)
process.CaloGeometryBuilder.SelectedCalos = ['HCAL']
process.ecalSimulationParametersEB.fromDD4hep = cms.bool(True)
process.ecalSimulationParametersEE.fromDD4hep = cms.bool(True)
process.ecalSimulationParametersES.fromDD4hep = cms.bool(True)
process.hcalSimulationParameters.fromDD4hep = cms.bool(True)

process.p1 = cms.Path(process.hcalCellParameterDump)
