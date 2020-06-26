import FWCore.ParameterSet.Config as cms

#
#  FIXME most probably this part is not needed for dispaced muons...
#

from  RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
duplicateDisplaceTrackCandidatesChi2Est = _Chi2MeasurementEstimator.clone(
    ComponentName = 'duplicateDisplacedTrackCandidatesChi2Est',
    MaxChi2 = 100
)

#for displaced global muons                                      
duplicateDisplacedTrackCandidates = DuplicateTrackMerger.clone(
    source='preDuplicateMergingDisplacedTracks',
    useInnermostState  = True,
    ttrhBuilderName    = 'WithAngleAndTemplate',
    chi2EstimatorName = 'duplicateDisplacedTrackCandidatesChi2Est'
)

#for displaced global muons
mergedDuplicateDisplacedTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'duplicateDisplacedTrackCandidates:candidates'
)

#for displaced global muons
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
duplicateDisplacedTrackClassifier = TrackCutClassifier.clone(
    src = 'mergedDuplicateDisplacedTracks',
    mva = dict(
	minPixelHits = [0,0,0],
	maxChi2 = [9999.,9999.,9999.],
	maxChi2n = [9999.,9999.,9999.],
	minLayers = [0,0,0],
	min3DLayers = [0,0,0],
	maxLostLayers = [99,99,99])
)

#for displaced global muons
displacedTracks = DuplicateListMerger.clone(
    originalSource      = 'preDuplicateMergingDisplacedTracks',
    originalMVAVals     = 'preDuplicateMergingDisplacedTracks:MVAValues',
    mergedSource        = 'mergedDuplicateDisplacedTracks',
    mergedMVAVals       = 'duplicateDisplacedTrackClassifier:MVAValues',
    candidateSource     = 'duplicateDisplacedTrackCandidates:candidates',
    candidateComponents = 'duplicateDisplacedTrackCandidates:candidateMap'
)

#for displaced global muons
displacedTracksTask = cms.Task(
    duplicateDisplacedTrackCandidates,
    mergedDuplicateDisplacedTracks,
    duplicateDisplacedTrackClassifier,
    displacedTracks
)
displacedTracksSequence = cms.Sequence(displacedTracksTask)
