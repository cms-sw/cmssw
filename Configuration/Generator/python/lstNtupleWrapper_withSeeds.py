import FWCore.ParameterSet.Config as cms

def customise(process):
    from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple, extendedContent
    process = customiseTrackingNtuple(process)
    process = extendedContent(process)
    
    if hasattr(process, "trackingNtuple"):
        # LEAVE SEEDS ON for the LST standalone code!
        process.trackingNtuple.includeSeeds = cms.untracked.bool(True)
        # Fix the CMSSW 16 Bug
        process.trackingNtuple.clusterMasks = cms.untracked.VPSet()
        
    return process
