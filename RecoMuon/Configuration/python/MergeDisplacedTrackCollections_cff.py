import FWCore.ParameterSet.Config as cms

#
#  FIXME most probably this part is not needed for dispaced muons...
#

from  RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
duplicateDisplaceTrackCandidatesChi2Est = _Chi2MeasurementEstimator.clone(
    ComponentName = "duplicateDisplacedTrackCandidatesChi2Est",
    MaxChi2 = 100,
)

#for displaced global muons                                      
duplicateDisplacedTrackCandidates = DuplicateTrackMerger.clone(
    source=cms.InputTag("preDuplicateMergingDisplacedTracks"),
    useInnermostState  = cms.bool(True),
    ttrhBuilderName    = cms.string("WithAngleAndTemplate"),
    chi2EstimatorName = "duplicateDisplacedTrackCandidatesChi2Est"
    )
#for displaced global muons
mergedDuplicateDisplacedTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("duplicateDisplacedTrackCandidates","candidates"),
    )


#for displaced global muons
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
duplicateDisplacedTrackClassifier = TrackCutClassifier.clone()
duplicateDisplacedTrackClassifier.src='mergedDuplicateDisplacedTracks'
duplicateDisplacedTrackClassifier.mva.minPixelHits = [0,0,0]
duplicateDisplacedTrackClassifier.mva.maxChi2 = [9999.,9999.,9999.]
duplicateDisplacedTrackClassifier.mva.maxChi2n = [9999.,9999.,9999.]
duplicateDisplacedTrackClassifier.mva.minLayers = [0,0,0]
duplicateDisplacedTrackClassifier.mva.min3DLayers = [0,0,0]
duplicateDisplacedTrackClassifier.mva.maxLostLayers = [99,99,99]


#for displaced global muons
displacedTracks = DuplicateListMerger.clone(
    originalSource = cms.InputTag("preDuplicateMergingDisplacedTracks"),
    originalMVAVals = cms.InputTag("preDuplicateMergingDisplacedTracks","MVAValues"),
    mergedSource = cms.InputTag("mergedDuplicateDisplacedTracks"),
    mergedMVAVals = cms.InputTag("duplicateDisplacedTrackClassifier","MVAValues"),
    candidateSource = cms.InputTag("duplicateDisplacedTrackCandidates","candidates"),
    candidateComponents = cms.InputTag("duplicateDisplacedTrackCandidates","candidateMap")
    )
#for displaced global muons
displacedTracksTask = cms.Task(
    duplicateDisplacedTrackCandidates,
    mergedDuplicateDisplacedTracks,
    duplicateDisplacedTrackClassifier,
    displacedTracks
    )
displacedTracksSequence = cms.Sequence(displacedTracksTask)

# This customization will be removed once we get the templates for
# phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(duplicateDisplacedTrackCandidates, ttrhBuilderName = "WithTrackAngle") # FIXME
