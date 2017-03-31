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
        '/store/relval/CMSSW_8_0_26/RelValTTbar_13/MINIAODSIM/80X_mcRun2_asymptotic_2016_TrancheIV_v8_width1p9mum_BS1p9-v1/00000/4C60817A-D5F3-E611-AB04-0025905B858A.root',
    ),
)

process.options = cms.untracked.PSet()

process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")

process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_2013_I1224539_DIJET')
process.rivetAnalyzer.OutputFile = cms.string('mcfile.yoda')

process.p = cms.Path(process.mergedGenParticles*process.genParticles2HepMC*process.rivetAnalyzer)
