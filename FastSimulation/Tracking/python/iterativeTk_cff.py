
##############################
# FastSim equivalent of RecoTracker/IterativeTracking/python/iterativeTk_cff.py
##############################

import FWCore.ParameterSet.Config as cms
from TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff import *
from FastSimulation.Tracking.InitialStep_cff import *
from FastSimulation.Tracking.DetachedTripletStep_cff import *
from FastSimulation.Tracking.LowPtTripletStep_cff import *
from FastSimulation.Tracking.PixelPairStep_cff import *
from FastSimulation.Tracking.MixedTripletStep_cff import *
from FastSimulation.Tracking.PixelLessStep_cff import *
from FastSimulation.Tracking.TobTecStep_cff import *
# the following loads a dummy empty track collection
# such that FastSim can import earlyGeneralTracks_cfi from full tracking
# todo: actual implementation of JetCore iteration  
from FastSimulation.Tracking.JetCoreRegionalStep_cff import *

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

