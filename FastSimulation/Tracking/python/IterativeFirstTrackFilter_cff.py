import FWCore.ParameterSet.Config as cms

# Track filtering and quality.
#   input:    iterativeFirstTrackMerging 
#   output:   generalTracks
#   sequence: iterativeFirstTrackFiltering
# Official sequence has loose and tight quality tracks, not reproduced
# here. (People will use generalTracks, eventually.)
from RecoTracker.IterativeTracking.FirstFilter_cfi import *
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
iterativeFirstTrackFiltering = cms.Sequence(firstStepTracksWithQuality+firstfilter)
firstStepTracksWithQuality.src = 'iterativeFirstTrackMerging'
firstStepTracksWithQuality.keepAllTracks = True
firstStepTracksWithQuality.copyExtras = True
firstStepTracksWithQuality.copyTrajectories = True

