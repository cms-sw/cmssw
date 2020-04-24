import FWCore.ParameterSet.Config as cms

process = cms.Process("dumpHepMC")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:gen.root')
)

process.dummy = cms.EDAnalyzer("DummyHepMCAnalyzer",
    src = cms.InputTag("VtxSmeared"),
    dumpHepMC = cms.untracked.bool(True),
    dumpPDF = cms.untracked.bool(False),
    checkPDG = cms.untracked.bool(False)
)

process.p = cms.Path(process.dummy)


