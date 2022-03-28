import FWCore.ParameterSet.Config as cms

def customise(process):
    
    if not hasattr(process, "options"):
        process.options = cms.untracked.PSet()
        
    process.options.numberOfThreads = 8
    process.options.numberOfStreams = 0
    
    return(process)
