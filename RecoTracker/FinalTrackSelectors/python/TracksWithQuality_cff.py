import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.FinalTrackSelectors.selectLoose_cfi import *
# Track filtering and quality.
#   input:    preFilterCmsTracks
#   output:   generalTracks
#   sequence: tracksWithQuality
withLooseQuality = copy.deepcopy(selectLoose)
import copy
from RecoTracker.FinalTrackSelectors.selectTight_cfi import *
withTightQuality = copy.deepcopy(selectTight)
import copy
from RecoTracker.FinalTrackSelectors.selectHighPurity_cfi import *
generalTracks = copy.deepcopy(selectHighPurity)
tracksWithQuality = cms.Sequence(withLooseQuality*withTightQuality*generalTracks)
withLooseQuality.src = 'preFilterCmsTracks'
withLooseQuality.keepAllTracks = False ## we only keep hthose who pass the filter

withLooseQuality.copyExtras = True
withLooseQuality.copyTrajectories = True
withTightQuality.src = 'withLooseQuality'
withTightQuality.keepAllTracks = True
withTightQuality.copyExtras = True
withTightQuality.copyTrajectories = True
generalTracks.src = 'withTightQuality'
generalTracks.keepAllTracks = True
generalTracks.copyExtras = True
generalTracks.copyTrajectories = True

