import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
            #relValTTbar
            '/store/relval/CMSSW_11_0_0/RelValTTbar_14TeV/GEN-SIM/110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/22BAADDB-EE84-794F-9A5D-812F341D8075.root'
        )
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
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cfi")
process.load("GeneratorInterface.RivetInterface.particleLevel_cfi")

process.path = cms.Path(process.genParticles2HepMC*process.particleLevel)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("particleLevel.root"),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep *_genParticles_*_*",
        "keep *_particleLevel_*_*",
    ),
)
process.outPath = cms.EndPath(process.out)
