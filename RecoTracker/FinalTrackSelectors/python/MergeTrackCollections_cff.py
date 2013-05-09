import FWCore.ParameterSet.Config as cms


import RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi

duplicateTrackCandidates = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateTrackMerger.clone(
    source=cms.InputTag("preDuplicateMergingGeneralTracks"),
    useInnermostState  = cms.bool(True),
    ttrhBuilderName    = cms.string("WithAngleAndTemplate")
    )
                                      
import RecoTracker.TrackProducer.TrackProducer_cfi
mergedDuplicateTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("duplicateTrackCandidates","candidates"),
    )

generalTracks = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingGeneralTracks"),
    mergedSource = cms.InputTag("mergedDuplicateTracks"),
    candidateSource = cms.InputTag("duplicateTrackCandidates","candidateMap")
    )


generalTracksSequence = cms.Sequence(
    duplicateTrackCandidates*
    mergedDuplicateTracks*
    generalTracks
    )

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
