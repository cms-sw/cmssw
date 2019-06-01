import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.load("GeneratorInterface.RivetInterface.mergedGenParticles_cfi")
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cff")
process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_10_6_0/RelValTTbar_13/MINIAODSIM/PUpmx25ns_106X_upgrade2018_realistic_v4-v1/10000/DB4F12C2-3B53-4247-A1C3-BEB74F177362.root',
    ),
)

process.options = cms.untracked.PSet()

process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")

process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_2013_I1224539_DIJET', 'MC_TTBAR', 'CMS_2018_I1662081')
process.rivetAnalyzer.OutputFile = cms.string('mcfile.yoda')

process.p = cms.Path(process.mergedGenParticles*process.genParticles2HepMC*process.rivetAnalyzer)
