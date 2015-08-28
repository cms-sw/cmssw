import FWCore.ParameterSet.Config as cms

def customise(process):
    
    if not hasattr(process, "options"):
        process.options = cms.untracked.PSet()
        
    process.options.numberOfThreads = cms.untracked.uint32(1)
    process.options.numberOfStreams = cms.untracked.uint32(1)
    
    return(process)
