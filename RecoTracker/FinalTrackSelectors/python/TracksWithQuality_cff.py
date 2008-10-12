import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

zeroStepWithLooseQuality = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
zeroStepWithTightQuality = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
zeroStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()


zeroStepWithLooseQuality.src = 'preFilterZeroStepTracks'
zeroStepWithLooseQuality.keepAllTracks = False ## we only keep those who pass the filter
zeroStepWithLooseQuality.copyExtras = True
zeroStepWithLooseQuality.copyTrajectories = True

zeroStepWithTightQuality.src = 'zeroStepWithLooseQuality'
zeroStepWithTightQuality.keepAllTracks = True
zeroStepWithTightQuality.copyExtras = True
zeroStepWithTightQuality.copyTrajectories = True

zeroStepTracksWithQuality.src = 'zeroStepWithTightQuality'
zeroStepTracksWithQuality.keepAllTracks = True
zeroStepTracksWithQuality.copyExtras = True
zeroStepTracksWithQuality.copyTrajectories = True


firstStepWithLooseQuality = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
firstStepWithTightQuality = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
preMergingFirstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()


firstStepWithLooseQuality.src = 'preFilterStepOneTracks'
firstStepWithLooseQuality.keepAllTracks = False ## we only keep those who pass the filter
firstStepWithLooseQuality.copyExtras = True
firstStepWithLooseQuality.copyTrajectories = True

firstStepWithTightQuality.src = 'firstStepWithLooseQuality'
firstStepWithTightQuality.keepAllTracks = True
firstStepWithTightQuality.copyExtras = True
firstStepWithTightQuality.copyTrajectories = True

preMergingFirstStepTracksWithQuality.src = 'firstStepWithTightQuality'
preMergingFirstStepTracksWithQuality.keepAllTracks = True
preMergingFirstStepTracksWithQuality.copyExtras = True
preMergingFirstStepTracksWithQuality.copyTrajectories = True

# Track filtering and quality.
#   input:    preFilterZeroStepTracks
#   output:   zeroStepTracksWithQuality
#   sequence: tracksWithQualityZeroStep
tracksWithQualityZeroStep = cms.Sequence(zeroStepWithLooseQuality*zeroStepWithTightQuality*zeroStepTracksWithQuality)


# Track filtering and quality.
#   input:    preFilterStepOneTracks
#   output:   firstStepTracksWithQuality
#   sequence: tracksWithQuality
tracksWithQualityStepOne = cms.Sequence(firstStepWithLooseQuality*firstStepWithTightQuality*preMergingFirstStepTracksWithQuality)


