import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

zeroStepWithLooseQuality = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'preFilterZeroStepTracks',
    keepAllTracks = False, ## we only keep those who pass the filter
    copyExtras = False,
    copyTrajectories = True
)

zeroStepWithTightQuality = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'zeroStepWithLooseQuality',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True
    )

zeroStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'zeroStepWithTightQuality',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True
    )


firstStepWithLooseQuality = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'preFilterStepOneTracks',
    keepAllTracks = False, ## we only keep those who pass the filter
    copyExtras = False,
    copyTrajectories = True
    )

firstStepWithTightQuality = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'firstStepWithLooseQuality',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True
    )
preMergingFirstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'firstStepWithTightQuality',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True
    )

# Track filtering and quality.
#   input:    preFilterZeroStepTracks
#   output:   zeroStepTracksWithQuality
#   sequence: tracksWithQualityZeroStep
tracksWithQualityZeroStepTask = cms.Task(zeroStepWithLooseQuality, zeroStepWithTightQuality, zeroStepTracksWithQuality)
tracksWithQualityZeroStep = cms.Sequence(tracksWithQualityZeroStepTask)

# Track filtering and quality.
#   input:    preFilterStepOneTracks
#   output:   firstStepTracksWithQuality
#   sequence: tracksWithQuality
tracksWithQualityStepOneTask = cms.Task(firstStepWithLooseQuality, firstStepWithTightQuality, preMergingFirstStepTracksWithQuality)
tracksWithQualityStepOne = cms.Sequence(tracksWithQualityStepOneTask)

