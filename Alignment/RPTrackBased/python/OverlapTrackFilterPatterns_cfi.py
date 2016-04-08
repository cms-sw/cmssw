import FWCore.ParameterSet.Config as cms

OverlapTrackFilterPatterns = cms.EDFilter("OverlapTrackFilterPatterns",
    # description of the track-candidate producer module
    #   use `RPSinglTrackCandFind' for parallel finder
    #   use `NonParallelTrackFinder' for non-parallel finder    
    tagRecognizedPatterns = cms.InputTag('NonParallelTrackFinder'),
    
    prescale_vvv = cms.uint32(5000),
    prescale_vvh = cms.uint32(1)
)
