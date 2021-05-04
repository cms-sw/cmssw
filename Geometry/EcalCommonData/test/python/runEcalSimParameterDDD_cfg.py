import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSimulationParametersTest")

process.load('Geometry.EcalCommonData.EcalOnly_cfi')
process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalGeom=dict()
    process.MessageLogger.EcalSim=dict()

process.load('Geometry.EcalCommonData.ecalSimulationParametersAnalyzer_cff')

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.ecalSimulationParametersAnalyzerEB+process.ecalSimulationParametersAnalyzerEE+process.ecalSimulationParametersAnalyzerES)
