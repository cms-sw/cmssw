import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.FinalTrackSelectors.selectLoose_cfi import *
# Track filtering and quality.
#   input:    preFilterFirstStepTracks
#   output:   firstStepTracksWithQuality
#   sequence: tracksWithQuality
withLooseQuality = copy.deepcopy(selectLoose)
import copy
from RecoTracker.FinalTrackSelectors.selectTight_cfi import *
withTightQuality = copy.deepcopy(selectTight)
import copy
from RecoTracker.FinalTrackSelectors.selectHighPurity_cfi import *
firstStepTracksWithQuality = copy.deepcopy(selectHighPurity)
tracksWithQuality = cms.Sequence(withLooseQuality*withTightQuality*firstStepTracksWithQuality)
withLooseQuality.src = 'preFilterFirstStepTracks'
withLooseQuality.keepAllTracks = False ## we only keep hthose who pass the filter

withLooseQuality.copyExtras = True
withLooseQuality.copyTrajectories = True
withTightQuality.src = 'withLooseQuality'
withTightQuality.keepAllTracks = True
withTightQuality.copyExtras = True
withTightQuality.copyTrajectories = True
firstStepTracksWithQuality.src = 'withTightQuality'
firstStepTracksWithQuality.keepAllTracks = True
firstStepTracksWithQuality.copyExtras = True
firstStepTracksWithQuality.copyTrajectories = True

