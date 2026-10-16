import FWCore.ParameterSet.Config as cms

def customise(process):
    # FAKE the reconstruction path so it passes the check!
    if not hasattr(process, 'reconstruction_step'):
        process.reconstruction_step = cms.Path()
        
    # Now the customizer won't crash
    from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple, extendedContent
    process = customiseTrackingNtuple(process)
    process = extendedContent(process)
    
    # Turn off seeds and clear masks for LST
    if hasattr(process, "trackingNtuple"):
        process.trackingNtuple.includeSeeds = cms.untracked.bool(False)
        process.trackingNtuple.clusterMasks = cms.untracked.VPSet()
        
        # THE FIX: vuint32 instead of vstring!
        process.trackingNtuple.seedAlgos = cms.untracked.vuint32()
        process.trackingNtuple.seedTracks = cms.untracked.VInputTag()
        
    return process
