import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
            #UL18 MiniAODv2
            '/store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/04A0B676-D63A-6D41-B47F-F4CF8CBE7DB8.root'
        )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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

# Turn on to write HepMC file
#process.genParticles2HepMC.writeHepMC = cms.untracked.bool(True)

process.path = cms.Path(process.mergedGenParticles*process.genParticles2HepMC)

