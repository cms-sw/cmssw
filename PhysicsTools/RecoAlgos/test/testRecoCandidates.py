from FWCore.ParameterSet.Config import *

process = Process("TestRecoCandidates")

process.include( "FWCore/MessageLogger/data/MessageLogger.cfi" )
process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi")
process.include( "PhysicsTools/RecoCandAlgos/data/allTrackCandidates.cfi")
process.include( "PhysicsTools/RecoCandAlgos/data/allMuonTrackCandidates.cfi")
process.include( "PhysicsTools/RecoCandAlgos/data/allStandAloneMuonTrackCandidates.cfi")
process.include( "PhysicsTools/RecoCandAlgos/data/allElectronTrackCandidates.cfi")
process.include( "PhysicsTools/RecoCandAlgos/data/allSuperClusterCandidates.cfi")

process.maxEvents = untracked.PSet( input = untracked.int32(1000) )

process.source = Source( "PoolSource",
  fileNames = untracked.vstring( "file:ZMM.root" )
)

process.out = OutputModule( "PoolOutputModule",
  fileName = untracked.string( "recocands.root" ),
  outputCommands= untracked.vstring(
    "drop *",
    "keep *_allTrackCandidates_*_*",
    "keep *_allMuonTrackCandidates_*_*",
    "keep *_allStandAloneMuonTrackCandidates_*_*",
    "keep *_allElectronTrackCandidates_*_*",
    "keep *_allSuperClusterCandidates_*_*"
  )
)
  
process.printEventNumber = OutputModule( "AsciiOutputModule" )
  
process.p = Path( 
  process.allTrackCandidates *
  process.allMuonTrackCandidates *
  process.allStandAloneMuonTrackCandidates *
  process.allElectronTrackCandidates *
  process.allSuperClusterCandidates 
)

process.o = EndPath( 
  process.out * 
  process.printEventNumber 
)
