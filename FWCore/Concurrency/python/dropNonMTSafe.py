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
  if ( x for x in process.producers.itervalues() if x.type_() == "OscarProducer" ) :
    from SimG4Core.Application.customiseMultithreadedSim import customiseMultithreadedSim
    customiseMultithreadedSim(process)

  return process
