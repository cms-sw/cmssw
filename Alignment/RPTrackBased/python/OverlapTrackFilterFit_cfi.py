import FWCore.ParameterSet.Config as cms

OverlapTrackFilterFit = cms.EDFilter("OverlapTrackFilterFit",
    # description of the track-candidate producer module
    #   use `RPSinglTrackCandFind' for parallel finder
    #   use `NonParallelTrackFinder' for non-parallel finder    
    tagRecognizedPatterns = cms.InputTag('NonParallelTrackFinder'),

    # z for calculating intercepts, in mm
    # absolute value, for arm=0 (1) minus (plus) sign is applied
    z0_abs = cms.double(217000),

    fitter = cms.PSet(
        verbosity = cms.untracked.uint32(0),
        maxResidualToSigma = cms.double(3),
        minimumHitsPerProjectionPerRP = cms.uint32(4)
    ),

    # thresholds for considering |x| or |y| large, in mm
    large_threshold_x = cms.double(5),
    large_threshold_y = cms.double(20),
    
    prescale_vvv = cms.uint32(5000),
    prescale_vvh = cms.uint32(1),
    prescale_hh = cms.uint32(1),
    prescale_lx = cms.uint32(1),
    prescale_ly = cms.uint32(1),
)
