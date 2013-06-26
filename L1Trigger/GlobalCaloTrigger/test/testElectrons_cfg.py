import FWCore.ParameterSet.Config as cms

process = cms.Process("TestGct")
process.load("L1Trigger.GlobalCaloTrigger.test.gctTest_cff")
process.load("L1Trigger.GlobalCaloTrigger.test.gctConfig_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p1 = cms.Path(process.gctemu)
process.gctemu.doElectrons = True
process.gctemu.inputFile = 'data/testEmDummy_'

