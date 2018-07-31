import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # ttbar pythia
        '/store/mc/RunIISummer16MiniAODv2/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/FED04F50-A3BE-E611-A893-0025900E3502.root',
        # ttbar herwig
        '/store/mc/RunIISummer16MiniAODv2/TT_TuneEE5C_13TeV-powheg-herwigpp/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/70000/FCCEE0A4-48B5-E611-9233-0023AEEEB538.root',
    ),
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
## configure process options
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    wantSummary      = cms.untracked.bool(True)
)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("GeneratorInterface.RivetInterface.mergedGenParticles_cfi")
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cfi")
process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")
process.load("GeneratorInterface.RivetInterface.particleLevel_cfi")
process.particleLevel.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")

process.path = cms.Path(process.mergedGenParticles*process.genParticles2HepMC*process.particleLevel)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("particleLevel.root"),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep *_genParticles_*_*",
        "keep *_particleLevel_*_*",
    ),
)
process.outPath = cms.EndPath(process.out)
