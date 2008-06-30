import FWCore.ParameterSet.Config as cms

process = cms.Process("TestGenParticlePruner")

process.include( "FWCore/MessageLogger/data/MessageLogger.cfi" )
process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi" )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring("file:genevents.root")
)

process.out = cms.OutputModule( "PoolOutputModule",
  fileName = cms.untracked.string( "genevents_pruned.root" ),
  outputCommands= cms.untracked.vstring(
    "drop *",
    "keep *_genParticles_*_*"
  )
)

process.prunedGenParticles = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = {Z0}",
    "drop pdgId = {Z0} & status = 2"
    )
)

process.printPrunedTree = cms.EDAnalyzer(
    "ParticleTreeDrawer",
    src = cms.InputTag("prunedGenParticles"),
    printIndex = cms.untracked.bool(True),
    printStatus = cms.untracked.bool(True)
)

process.p = cms.Path(
    process.prunedGenParticles *
    process.printPrunedTree
)

process.o = cms.EndPath(
    process.out 
)
