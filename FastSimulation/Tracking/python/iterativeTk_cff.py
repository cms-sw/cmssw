
##############################
# FastSim equivalent of RecoTracker/IterativeTracking/python/iterativeTk_cff.py
##############################

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff import *
#importing iterations directly from RecoTracker
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
# the following loads a dummy empty track collection
# such that FastSim can import earlyGeneralTracks_cfi from full tracking
# todo: actual implementation of JetCore iteration  
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import *

import RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi
# todo, import MuonSeededStep_cff, preDuplicateMergingGeneralTracks_cfi, MergeTrackCollections_cff, ConversionStep_cff

generalTracksBeforeMixing = RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi.earlyGeneralTracks.clone()

iterTracking = cms.Sequence(
    InitialStep
    +DetachedTripletStep
    +LowPtTripletStep
    +PixelPairStep
    +MixedTripletStep
    +PixelLessStep
    +TobTecStep
    +JetCoreRegionalStep
    +generalTracksBeforeMixing)

_iterTracking_Phase1 = cms.Sequence(
    InitialStep
    +LowPtQuadStep
    +HighPtTripletStep
    +LowPtTripletStep
    +DetachedQuadStep
    +DetachedTripletStep
    +PixelPairStep
    +MixedTripletStep
    +PixelLessStep
    +TobTecStep
    +JetCoreRegionalStep
    +generalTracksBeforeMixing)

trackingPhase1.toReplaceWith(iterTracking, _iterTracking_Phase1)

