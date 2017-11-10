import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi import *
from RecoTracker.FinalTrackSelectors.DuplicateListMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
duplicateTrackCandidatesChi2Est = _Chi2MeasurementEstimator.clone(
    ComponentName = "duplicateTrackCandidatesChi2Est",
    MaxChi2 = 100,
)

duplicateTrackCandidates = DuplicateTrackMerger.clone()
duplicateTrackCandidates.source = cms.InputTag("preDuplicateMergingGeneralTracks")
duplicateTrackCandidates.useInnermostState  = True
duplicateTrackCandidates.ttrhBuilderName   = "WithAngleAndTemplate"
duplicateTrackCandidates.chi2EstimatorName = "duplicateTrackCandidatesChi2Est"
                                     
import RecoTracker.TrackProducer.TrackProducer_cfi
mergedDuplicateTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
mergedDuplicateTracks.src = cms.InputTag("duplicateTrackCandidates","candidates")
mergedDuplicateTracks.Fitter='RKFittingSmoother' # no outlier rejection!

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
duplicateTrackClassifier = TrackCutClassifier.clone()
duplicateTrackClassifier.src='mergedDuplicateTracks'
duplicateTrackClassifier.mva.minPixelHits = [0,0,0]
duplicateTrackClassifier.mva.maxChi2 = [9999.,9999.,9999.]
duplicateTrackClassifier.mva.maxChi2n = [10.,1.0,0.4]  # [9999.,9999.,9999.]
duplicateTrackClassifier.mva.minLayers = [0,0,0]
duplicateTrackClassifier.mva.min3DLayers = [0,0,0]
duplicateTrackClassifier.mva.maxLostLayers = [99,99,99]
# This customization will be removed once we get the templates for
# phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(duplicateTrackCandidates, ttrhBuilderName = "WithTrackAngle") # FIXME

generalTracks = DuplicateListMerger.clone()
generalTracks.originalSource = cms.InputTag("preDuplicateMergingGeneralTracks")
generalTracks.originalMVAVals = cms.InputTag("preDuplicateMergingGeneralTracks","MVAValues")
generalTracks.mergedSource = cms.InputTag("mergedDuplicateTracks")
generalTracks.mergedMVAVals = cms.InputTag("duplicateTrackClassifier","MVAValues")
generalTracks.candidateSource = cms.InputTag("duplicateTrackCandidates","candidates")
generalTracks.candidateComponents = cms.InputTag("duplicateTrackCandidates","candidateMap")


generalTracksSequence = cms.Sequence(
    duplicateTrackCandidates*
    mergedDuplicateTracks*
    duplicateTrackClassifier*
    generalTracks
    )
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(generalTracksSequence, 
                      cms.Sequence(
        duplicateTrackCandidates*
        mergedDuplicateTracks*
        duplicateTrackClassifier
        )
)
def _fastSimGeneralTracks(process):
    from FastSimulation.Configuration.DigiAliases_cff import loadDigiAliasesWasCalled
    if loadDigiAliasesWasCalled:
        from FastSimulation.Configuration.DigiAliases_cff import generalTracks
        process.generalTracks = generalTracks
        return
    from Configuration.StandardSequences.Digi_cff import generalTracks
    process.generalTracks = generalTracks
modifyMergeTrackCollections_fastSimGeneralTracks = fastSim.makeProcessModifier( _fastSimGeneralTracks )

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
