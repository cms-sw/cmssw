import FWCore.ParameterSet.Config as cms

from  RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *

#for displaced global muons                                      
duplicateDisplacedTrackCandidates = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateTrackMerger.clone(
    source=cms.InputTag("preDuplicateMergingDisplacedTracks"),
    useInnermostState  = cms.bool(True),
    ttrhBuilderName    = cms.string("WithAngleAndTemplate")
    )
#for displaced global muons
mergedDuplicateDisplacedTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("duplicateDisplacedTrackCandidates","candidates"),
    )
#for displaced global muons
duplicateDisplacedTrackSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='mergedDuplicateDisplacedTracks',
    trackSelectors= cms.VPSet(
    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
    name = 'duplicateDisplacedTrackSelectorLoose',
    minHitsToBypassChecks = cms.uint32(0),
            ),
        )
    )

#for displaced global muons
displacedTracks = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingDisplacedTracks"),
    mergedSource = cms.InputTag("mergedDuplicateDisplacedTracks"),
    mergedMVAVals = cms.InputTag("duplicateDisplacedTrackSelector","MVAVals"),
    candidateSource = cms.InputTag("duplicateDisplacedTrackCandidates","candidateMap")
    )
#for displaced global muons
displacedTracksSequence = cms.Sequence(
    duplicateDisplacedTrackCandidates*
    mergedDuplicateDisplacedTracks*
    duplicateDisplacedTrackSelector*
    displacedTracks
    )
