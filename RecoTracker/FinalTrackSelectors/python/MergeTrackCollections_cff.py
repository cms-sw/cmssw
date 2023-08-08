import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi import *
from RecoTracker.FinalTrackSelectors.DuplicateListMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import SiPixelTemplateStoreESProducer

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
duplicateTrackCandidatesChi2Est = _Chi2MeasurementEstimator.clone(
    ComponentName = "duplicateTrackCandidatesChi2Est",
    MaxChi2 = 100,
)

duplicateTrackCandidates = DuplicateTrackMerger.clone(
    source = "preDuplicateMergingGeneralTracks",
    useInnermostState  = True,
    ttrhBuilderName   = "WithAngleAndTemplate",
    chi2EstimatorName = "duplicateTrackCandidatesChi2Est"
)

import RecoTracker.TrackProducer.TrackProducer_cfi
mergedDuplicateTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = "duplicateTrackCandidates:candidates",
    Fitter='RKFittingSmoother' # no outlier rejection!
)

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
duplicateTrackClassifier = TrackCutClassifier.clone(
    src='mergedDuplicateTracks',
    mva = dict(
	minPixelHits = [0,0,0],
	maxChi2 = [9999.,9999.,9999.],
	maxChi2n = [10.,1.0,0.4],  # [9999.,9999.,9999.]
	minLayers = [0,0,0],
	min3DLayers = [0,0,0],
	maxLostLayers = [99,99,99])
)

generalTracks = DuplicateListMerger.clone(
    originalSource      = "preDuplicateMergingGeneralTracks",
    originalMVAVals     = "preDuplicateMergingGeneralTracks:MVAValues",
    mergedSource        = "mergedDuplicateTracks",
    mergedMVAVals       = "duplicateTrackClassifier:MVAValues",
    candidateSource     = "duplicateTrackCandidates:candidates",
    candidateComponents = "duplicateTrackCandidates:candidateMap"
)

generalTracksTask = cms.Task(
    duplicateTrackCandidates,
    mergedDuplicateTracks,
    duplicateTrackClassifier,
    generalTracks,
    SiPixelTemplateStoreESProducer
    )
generalTracksSequence = cms.Sequence(generalTracksTask)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(generalTracksTask, 
                      cms.Task(duplicateTrackCandidates,
                               mergedDuplicateTracks,
                               duplicateTrackClassifier)
)

def _fastSimGeneralTracks(process):
    from FastSimulation.Configuration.DigiAliases_cff import loadGeneralTracksAlias
    loadGeneralTracksAlias(process)
modifyMergeTrackCollections_fastSimGeneralTracks = fastSim.makeProcessModifier( _fastSimGeneralTracks )

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
conversionStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers     = ['convStepTracks'],
    hasSelector        = [1],
    selectedTrackQuals = ['convStepSelector:convStep'],
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(1), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(mergedDuplicateTracks, TrajectoryInEvent = True)
phase2_timing_layer.toModify(generalTracks, copyTrajectories = True)
