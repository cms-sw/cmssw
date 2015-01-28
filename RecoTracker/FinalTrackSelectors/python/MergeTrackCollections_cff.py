import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi

duplicateTrackCandidates = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateTrackMerger.clone(
    source=cms.InputTag("preDuplicateMergingGeneralTracks"),
    useInnermostState  = cms.bool(True),
    ttrhBuilderName    = cms.string("WithAngleAndTemplate")
    )
#for displaced global muons                                      
duplicateDisplacedTrackCandidates = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateTrackMerger.clone(
    source=cms.InputTag("preDuplicateMergingDisplacedTracks"),
    useInnermostState  = cms.bool(True),
    ttrhBuilderName    = cms.string("WithAngleAndTemplate")
    )

import RecoTracker.TrackProducer.TrackProducer_cfi
mergedDuplicateTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("duplicateTrackCandidates","candidates"),
    )
#for displaced global muons
mergedDuplicateDisplacedTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("duplicateDisplacedTrackCandidates","candidates"),
    )
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
duplicateTrackSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='mergedDuplicateTracks',
    trackSelectors= cms.VPSet(
    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
    name = 'duplicateTrackSelectorLoose',
    minHitsToBypassChecks = cms.uint32(0),
            ),
        )
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

generalTracks = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingGeneralTracks"),
    mergedSource = cms.InputTag("mergedDuplicateTracks"),
    mergedMVAVals = cms.InputTag("duplicateTrackSelector","MVAVals"),
    candidateSource = cms.InputTag("duplicateTrackCandidates","candidateMap")
    )
#for displaced global muons
displacedTracks = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingDisplacedTracks"),
    mergedSource = cms.InputTag("mergedDuplicateDisplacedTracks"),
    mergedMVAVals = cms.InputTag("duplicateDisplacedTrackSelector","MVAVals"),
    candidateSource = cms.InputTag("duplicateDisplacedTrackCandidates","candidateMap")
    )

generalTracksSequence = cms.Sequence(
    duplicateTrackCandidates*
    mergedDuplicateTracks*
    duplicateTrackSelector*
    generalTracks
    )
#for displaced global muons
displacedTracksSequence = cms.Sequence(
    duplicateDisplacedTrackCandidates*
    mergedDuplicateDisplacedTracks*
    duplicateDisplacedTrackSelector*
    displacedTracks
    )
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
conversionStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('convStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("convStepSelector","convStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(1), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
