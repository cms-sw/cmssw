import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.Phase1_InitialStepPreSplitting_cff import *
from RecoTracker.IterativeTracking.Phase1_InitialStep_cff import *
from RecoTracker.IterativeTracking.Phase1_HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1_DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1_DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1_LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1_MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1_PixelLessStep_cff import *
from RecoTracker.IterativeTracking.Phase1_TobTecStep_cff import *
from RecoTracker.IterativeTracking.Phase1_JetCoreRegionalStep_cff import *

from RecoTracker.FinalTrackSelectors.Phase1_earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

iterTracking = cms.Sequence(
    InitialStepPreSplitting
    + InitialStep
    + HighPtTripletStep
    + DetachedQuadStep
#    + DetachedTripletStep # FIXME: dropped for time being, but it may be enabled in the course of tuning
    + LowPtQuadStep
    + LowPtTripletStep
    + MixedTripletStep
    + PixelLessStep
    + TobTecStep
    + JetCoreRegionalStep
    + earlyGeneralTracks
    + muonSeededStep
    + preDuplicateMergingGeneralTracks
    + generalTracksSequence
    + ConvStep
    + conversionStepTracks
)
