import FWCore.ParameterSet.Config as cms

def enableIMT(process):
  process.InitRootHandlers = cms.Service("InitRootHandlers",
      EnableIMT = cms.untracked.bool(True)
  )
  return process
