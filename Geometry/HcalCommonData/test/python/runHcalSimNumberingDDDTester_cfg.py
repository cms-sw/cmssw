import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("HcalSimNumberingTest",Run3)

#process.load('Geometry.HcalCommonData.testPhase2GeometryFine_cff')
#process.load('Geometry.HcalCommonData.hcalParameters_cff')
#process.load('Geometry.HcalCommonData.hcalSimulationParameters_cff')
process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalSimNumberingTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
