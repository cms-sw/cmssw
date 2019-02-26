import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import *
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *

from Configuration.StandardSequences.Eras import eras
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1


def customise_MVAPhase1(process):
	trackingPhase1.toReplaceWith(detachedQuadStep, TrackMVAClassifierDetached.clone(
		mva = dict(GBRForestLabel = 'MVASelectorDetachedQuadStep_Phase1'),
		src = 'detachedQuadStepTracks',
		qualityCuts = [-0.5,0.0,0.5]
	))

        trackingPhase1.toReplaceWith(detachedTripletStep, TrackMVAClassifierDetached.clone(
                mva = dict(GBRForestLabel = 'MVASelectorDetachedTripletStep_Phase1'),
                src = 'detachedTripletStepTracks',
                qualityCuts = [-0.2,0.3,0.8]
        ))

        trackingPhase1.toReplaceWith(highPtTripletStep, TrackMVAClassifierPrompt.clone(
                mva = dict(GBRForestLabel = 'MVASelectorHighPtTripletStep_Phase1'),
                src = 'highPtTripletStepTracks',
                qualityCuts = [0.2,0.3,0.4]
        ))

	trackingPhase1.toReplaceWith(initialStep, TrackMVAClassifierPrompt.clone(
		mva = dict(GBRForestLabel = 'MVASelectorInitialStep_Phase1'),
		src = 'initialStepTracks',
		qualityCuts = [-0.95,-0.85,-0.75]
	))

	trackingPhase1.toReplaceWith(jetCoreRegionalStep, TrackMVAClassifierPrompt.clone(
                mva = dict(GBRForestLabel = 'MVASelectorJetCoreRegionalStep_Phase1'),
                src = 'jetCoreRegionalStepTracks',
                qualityCuts = [-0.2,0.0,0.4]
        ))

        trackingPhase1.toReplaceWith(lowPtQuadStep, TrackMVAClassifierPrompt.clone(
                mva = dict(GBRForestLabel = 'MVASelectorLowPtQuadStep_Phase1'),
                src = 'lowPtQuadStepTracks',
                qualityCuts = [-0.7,-0.35,-0.15]
        ))

        trackingPhase1.toReplaceWith(lowPtTripletStep, TrackMVAClassifierPrompt.clone(
                mva = dict(GBRForestLabel = 'MVASelectorLowPtTripletStep_Phase1'),
                src = 'lowPtTripletStepTracks',
                qualityCuts = [-0.4,0.0,0.3]
        ))

	trackingPhase1.toReplaceWith(mixedTripletStep, TrackMVAClassifierDetached.clone(
                mva = dict(GBRForestLabel = 'MVASelectorMixedTripletStep_Phase1'),
                src = 'mixedTripletStepTracks',
                qualityCuts = [-0.5,0.0,0.5]
        ))

        trackingPhase1.toReplaceWith(pixelLessStep, TrackMVAClassifierDetached.clone(
                mva = dict(GBRForestLabel = 'MVASelectorPixelLessStep_Phase1'),
                src = 'pixelLessStepTracks',
                qualityCuts = [-0.4,0.0,0.4]
        ))

        trackingPhase1.toReplaceWith(pixelPairStep, TrackMVAClassifierDetached.clone(
                mva = dict(GBRForestLabel = 'MVASelectorPixelPairStep_Phase1'),
                src = 'pixelPairStepTracks',
                qualityCuts = [-0.2,0.0,0.3]
        ))

        trackingPhase1.toReplaceWith(tobTecStep, TrackMVAClassifierDetached.clone(
                mva = dict(GBRForestLabel = 'MVASelectorTobTecStep_Phase1'),
                src = 'tobTecStepTracks',
                qualityCuts = [-0.6,-0.45,-0.3]
        ))

	return process
