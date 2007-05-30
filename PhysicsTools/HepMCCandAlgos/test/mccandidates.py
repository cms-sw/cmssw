from FWCore.ParameterSet.Config import *

process = Process("MCCand")

process.include( "FWCore/MessageLogger/data/MessageLogger.cfi" )
process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genParticleCandidatesFast.cfi")
# The following is still not clear how should be converted to .py...
# process.include( "PhysicsTools/HepMCCandAlgos/test/h4l.cff")
# include "PhysicsTools/HepMCCandAlgos/test/ttbar.cff"
# include "PhysicsTools/HepMCCandAlgos/test/hpp.cff"
# include "PhysicsTools/HepMCCandAlgos/test/h4l.cff"
# include "PhysicsTools/HepMCCandAlgos/test/herwig.cff"

process.add_( Service("RandomNumberGeneratorService",
              sourceSeed= untracked.uint32( 123456789 ) ) )

process.maxEvents = untracked.PSet( input = untracked.int32(50) )

from PhysicsTools.HepMCCandAlgos.data.h4l_cff import pythiaSource

process.source = pythiaSource

process.out = OutputModule( "PoolOutputModule",
  fileName = untracked.string( "genevents.root" ),
  outputCommands= untracked.vstring(
    "drop *",
    "keep *_genParticleCandidates_*_*"
  )
)
  
process.printEventNumber = OutputModule( "AsciiOutputModule" )
  
process.p = Path( 
  process.genParticleCandidates 
)

process.o = EndPath( 
  process.out * 
  process.printEventNumber 
)
