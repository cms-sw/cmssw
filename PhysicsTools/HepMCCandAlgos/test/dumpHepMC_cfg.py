import FWCore.ParameterSet.Config as cms

process = cms.Process("dumpHepMC")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.load("PhysicsTools.HepMCCandAlgos.dummyHepMCAnalyzer_cfi")
process.dummyHepMCAnalyzer.dumpPDF = True
process.dummyHepMCAnalyzer.checkPDG = True

process.p = cms.Path(process.dummyHepMCAnalyzer)


# foo bar baz
# D6biK4DXOeEgL
# TtPnGm6updZ0T
