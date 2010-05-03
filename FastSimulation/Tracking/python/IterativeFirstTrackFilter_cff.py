import FWCore.ParameterSet.Config as cms

# Track filtering and quality.
#   input:    iterativeFirstTrackMerging 
#   output:   generalTracks
#   sequence: iterativeFirstTrackFiltering
# Official sequence has loose and tight quality tracks, not reproduced
# here. (People will use generalTracks, eventually.)
###from RecoTracker.IterativeTracking.FirstFilter_cfi import *


import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi


zeroStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
zeroStepTracksWithQuality.src = 'iterativeZeroTrackMerging'
zeroStepTracksWithQuality.keepAllTracks = True
zeroStepTracksWithQuality.copyExtras = True
zeroStepTracksWithQuality.copyTrajectories = True


firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
firstStepTracksWithQuality.src = 'iterativeFirstTrackMerging'
firstStepTracksWithQuality.keepAllTracks = True
firstStepTracksWithQuality.copyExtras = True
firstStepTracksWithQuality.copyTrajectories = True

zeroStepFilter = cms.EDProducer("QualityFilter",
     TrackQuality = cms.string('highPurity'),
     recTracks = cms.InputTag("zeroStepTracksWithQuality:")
)


firstfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("firstStepTracksWithQuality")
)


iterativeZeroTrackFiltering = cms.Sequence(zeroStepTracksWithQuality+zeroStepFilter)

iterativeFirstTrackFiltering = cms.Sequence(firstStepTracksWithQuality+firstfilter)
