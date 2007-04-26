import FWCore.ParameterSet.Config as cms

process = cms.Process("MCCand")

process.include( "FWCore/MessageLogger/data/MessageLogger.cfi" )
process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genParticleCandidatesFast.cfi")
# The following is still not clear how should be converted to .py...
# process.include( "PhysicsTools/HepMCCandAlgos/test/h4l.cff")
# include "PhysicsTools/HepMCCandAlgos/test/ttbar.cff"
# include "PhysicsTools/HepMCCandAlgos/test/hpp.cff"
# include "PhysicsTools/HepMCCandAlgos/test/h4l.cff"
# include "PhysicsTools/HepMCCandAlgos/test/herwig.cff"

process.add_(cms.Service("RandomNumberGeneratorService",
             sourceSeed= cms.untracked.uint32( 123456789 )))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50) )

from h4l import pythiaSource

process.source = pythiaSource

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("genevents.root"),
  outputCommands= cms.untracked.vstring(
    "drop *",
    "keep *_genParticleCandidates_*_*"
 )
)
  
process.printEventNumber = cms.OutputModule( "AsciiOutputModule" )
  
process.p = cms.Path( 
  process.genParticleCandidates 
)

process.o = cms.EndPath( 
  process.out * 
  process.printEventNumber 
)
