import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

process.load("GeneratorInterface.RivetInterface.mergedGenParticles_cfi")
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cff")
process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_9_4_5_cand1/RelValTTbar_13/MINIAODSIM/94X_mc2017_realistic_v14_PU_RelVal_rmaod-v1/10000/84A84D5B-9E2E-E811-B103-0CC47A7C35F4.root',
    ),
)

process.options = cms.untracked.PSet()

process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")

process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_2016_I1459051')
process.rivetAnalyzer.OutputFile = cms.string('mcfile.yoda')

process.p = cms.Path(process.mergedGenParticles*process.genParticles2HepMC*process.rivetAnalyzer)
