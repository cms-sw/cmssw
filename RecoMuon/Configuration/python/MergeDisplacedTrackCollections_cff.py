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
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
duplicateDisplacedTrackClassifier = TrackCutClassifier.clone()
duplicateDisplacedTrackClassifier.src='mergedDuplicateDisplacedTracks'
duplicateDisplacedTrackClassifier.mva.minPixelHits = [0,0,0]
duplicateDisplacedTrackClassifier.mva.maxChi2 = [9999.,9999.,9999.]
duplicateDisplacedTrackClassifier.mva.maxChi2n = [9999.,9999.,9999.]
duplicateDisplacedTrackClassifier.mva.minLayers = [0,0,0]
duplicateDisplacedTrackClassifier.mva.min3DLayers = [0,0,0]
duplicateDisplacedTrackClassifier.mva.maxLostLayers = [99,99,99]


#for displaced global muons
displacedTracks = RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi.duplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingDisplacedTracks"),
    mergedSource = cms.InputTag("mergedDuplicateDisplacedTracks"),
    mergedMVAVals = cms.InputTag("duplicateDisplacedTrackClassifier","MVAValues"),
    candidateSource = cms.InputTag("duplicateDisplacedTrackCandidates","candidateMap")
    )
#for displaced global muons
displacedTracksSequence = cms.Sequence(
    duplicateDisplacedTrackCandidates*
    mergedDuplicateDisplacedTracks*
    duplicateDisplacedTrackClassifier*
    displacedTracks
    )
