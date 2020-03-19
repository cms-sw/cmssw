import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalSimNumberingTest")

#process.load('Geometry.HcalCommonData.testPhase1GeometryXML_cfi')
#process.load('Geometry.HcalCommonData.hcalSimulationParameters_cff')
process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('HcalGeom')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalSimNumberingTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
