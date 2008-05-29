import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectLoose_cfi
# Track filtering and quality.
#   input:    preFilterCmsTracks
#   output:   generalTracks
#   sequence: tracksWithQuality
withLooseQuality = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
import RecoTracker.FinalTrackSelectors.selectTight_cfi
withTightQuality = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
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

