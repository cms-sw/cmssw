import FWCore.ParameterSet.Config as cms

def _dropFromPaths(process,name):
  if hasattr(process,name):
    m = getattr(process,name)
    for p in process.paths.itervalues():
      p.remove(m)
    delattr(process,name)
  
def dropNonMTSafe(process):
  process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(4),
                                       sizeOfStackForThreadsInKB = cms.untracked.uint32(10*1024),
                                       numberOfStreams = cms.untracked.uint32(0))
  if not hasattr(process,"ZombieKillerService"):
    process.add_(cms.Service("ZombieKillerService",
                             secondsBetweenChecks = cms.untracked.uint32(120),
                             numberOfAllowedFailedChecksInARow = cms.untracked.uint32(4)))

  return process
