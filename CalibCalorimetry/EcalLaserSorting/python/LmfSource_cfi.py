import FWCore.ParameterSet.Config as cms

source = cms.Source("LmfSource",
  fileNames = cms.vstring("in.lmf"),
  preScale  = cms.uint32(1),
  orderedRead = cms.bool(True),
  verbosity = cms.untracked.int32(0)
)
