import FWCore.ParameterSet.Config as cms

def _dropFromPaths(process,name):
  if hasattr(process,name):
    m = getattr(process,name)
    for p in process.paths.itervalues():
      p.remove(m)
    delattr(process,name)
  
def dropNonMTSafe(process):
  if hasattr(process, "DTDataIntegrityTask"): del process.DTDataIntegrityTask
  if hasattr(process, "FastTimerService"): del process.FastTimerService
  if hasattr(process, "SiStripDetInfoFileReader"): del process.SiStripDetInfoFileReader
  if hasattr(process, "TkDetMap"): del process.TkDetMap
  if hasattr(process, "DQM"): del process.DQM
  if hasattr(process, "PoolDBOutputService"): del process.PoolDBOutputService
  #drop items dependent on TkDetMap
  _dropFromPaths(process,"siStripFEDMonitor")
  _dropFromPaths(process,"SiStripMonitorDigi")
  _dropFromPaths(process,"SiStripMonitorCluster")
  _dropFromPaths(process,"SiStripMonitorTrack_ckf")
  _dropFromPaths(process,"SiStripMonitorClusterBPTX")
  _dropFromPaths(process,"siStripOfflineAnalyser")
  _dropFromPaths(process,"SiStripMonitorTrackCommon")
  _dropFromPaths(process,"SiStripMonitorTrack_hi")

  process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(4),
                                       sizeOfStackForThreadsInKB = cms.untracked.uint32(10*1024),
                                       numberOfStreams = cms.untracked.uint32(0))
                                       
  return process
