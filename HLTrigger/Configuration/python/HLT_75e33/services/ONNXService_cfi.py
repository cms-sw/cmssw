import FWCore.ParameterSet.Config as cms

ONNXService = cms.Service("ONNXService",
  allowTF32 = cms.bool(False),
  verbose = cms.untracked.bool(False)
)
