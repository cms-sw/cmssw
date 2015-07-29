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

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
duplicateTrackClassifier = TrackCutClassifier.clone()
duplicateTrackClassifier.src='mergedDuplicateTracks'
duplicateTrackClassifier.mva.minPixelHits = [0,0,0]
duplicateTrackClassifier.mva.maxChi2 = [9999.,9999.,9999.]
duplicateTrackClassifier.mva.maxChi2n = [9999.,9999.,9999.]
duplicateTrackClassifier.mva.minLayers = [0,0,0]
duplicateTrackClassifier.mva.min3DLayers = [0,0,0]
duplicateTrackClassifier.mva.maxLostLayers = [99,99,99]



generalTracks = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingGeneralTracks"),
    mergedSource = cms.InputTag("mergedDuplicateTracks"),
    mergedMVAVals = cms.InputTag("duplicateTrackClassifier","MVAValues"),
    candidateSource = cms.InputTag("duplicateTrackCandidates","candidateMap")
    )


generalTracksSequence = cms.Sequence(
    duplicateTrackCandidates*
    mergedDuplicateTracks*
    duplicateTrackClassifier*
    generalTracks
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
