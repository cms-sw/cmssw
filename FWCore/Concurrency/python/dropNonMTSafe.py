import FWCore.ParameterSet.Config as cms

def dropNonMTSafe(process):
  if hasattr(process, "DTDataIntegrityTask"): del process.DTDataIntegrityTask
  if hasattr(process, "FastTimerService"): del process.FastTimerService
  if hasattr(process, "SiStripDetInfoFileReader"): del process.SiStripDetInfoFileReader
  if hasattr(process, "TkDetMap"): del process.TkDetMap
  process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(8),
                                       numberOfStreams = cms.untracked.uint32(0))
  return process
