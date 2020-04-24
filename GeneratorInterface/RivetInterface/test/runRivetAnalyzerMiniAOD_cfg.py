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
        '/store/relval/CMSSW_9_1_0_pre1/RelValTTbar_13/MINIAODSIM/PU25ns_90X_mcRun2_asymptotic_v5-v1/00000/BE649FEB-C610-E711-AFD5-0CC47A4D769E.root',
    ),
)

process.options = cms.untracked.PSet()

process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")

process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_2013_I1224539_DIJET')
process.rivetAnalyzer.OutputFile = cms.string('mcfile.yoda')

process.p = cms.Path(process.mergedGenParticles*process.genParticles2HepMC*process.rivetAnalyzer)
