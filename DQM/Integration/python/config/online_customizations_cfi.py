import FWCore.ParameterSet.Config as cms

def customise(process):
    
    if not hasattr(process, "options"):
        process.options = cms.untracked.PSet()
        
    process.options.numberOfThreads = 1
    process.options.numberOfStreams = 1
    
    return(process)
