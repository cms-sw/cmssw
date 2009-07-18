import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
    skipBadFiles = cms.untracked.bool(True),                        
    inputCommands = cms.untracked.vstring('drop *', 'keep *_source_*_HLT'),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
)

