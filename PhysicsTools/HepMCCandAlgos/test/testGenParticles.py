from FWCore.ParameterSet.Config import *

process = Process("testGenParticles")

process.include( "FWCore/MessageLogger/data/MessageLogger.cfi" )
process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genParticleCandidates.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genParticles.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genEventWeight.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genEventScale.cfi")

process.add_( Service("RandomNumberGeneratorService",
              sourceSeed= untracked.uint32( 123456789 ) ) )

process.maxEvents = untracked.PSet( input = untracked.int32(1000) )

from PhysicsTools.HepMCCandAlgos.data.h4l_cff import pythiaSource

process.source = pythiaSource

process.testGenParticles = EDAnalyzer( "TestGenParticleCandidates",
  src = InputTag("genParticleCandidates")
)
  
process.out = OutputModule( "PoolOutputModule",
  fileName = untracked.string( "genevents.root" ),
  outputCommands= untracked.vstring(
    "drop *",
    "keep *_genParticleCandidates_*_*",
    "keep *_genParticles_*_*",
    "keep *_genEventWeight_*_*"
  )
)
  
process.p = Path( 
  process.genParticleCandidates *
  process.genParticles *
  process.genEventWeight *
  process.testGenParticles
)

process.o = EndPath( 
  process.out 
)
