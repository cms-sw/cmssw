import FWCore.ParameterSet.Config as cms

# Track filtering and quality.
#   input:    iterativeFirstTrackMerging 
#   output:   generalTracks
#   sequence: iterativeFirstTrackFiltering
# Official sequence has loose and tight quality tracks, not reproduced
# here. (People will use generalTracks, eventually.)
from RecoTracker.IterativeTracking.FirstFilter_cfi import *
import copy
from RecoTracker.FinalTrackSelectors.selectHighPurity_cfi import *
generalTracks = copy.deepcopy(selectHighPurity)
iterativeFirstTrackFiltering = cms.Sequence(generalTracks+firstfilter)
generalTracks.src = 'iterativeFirstTrackMerging'
generalTracks.keepAllTracks = True
generalTracks.copyExtras = True
generalTracks.copyTrajectories = True

