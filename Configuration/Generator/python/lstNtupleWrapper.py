import FWCore.ParameterSet.Config as cms

def customise(process):
    from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple, extendedContent
    process = customiseTrackingNtuple(process)
    process = extendedContent(process)
    
    if hasattr(process, "trackingNtuple"):
        # ALL THE HARD-WON C++ TYPE FIXES!
        process.trackingNtuple.includeSeeds = cms.untracked.bool(False)
        process.trackingNtuple.clusterMasks = cms.untracked.VPSet()
        process.trackingNtuple.seedAlgos = cms.untracked.vuint32()
        process.trackingNtuple.seedTracks = cms.untracked.VInputTag()
        
    return process
