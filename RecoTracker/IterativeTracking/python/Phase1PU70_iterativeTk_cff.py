import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.Phase1PU70_InitialStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_PixelPairStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_TobTecStep_cff import *
from RecoTracker.FinalTrackSelectors.Phase1PU70_earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.Phase1PU70_MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.Phase1PU70_preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.Phase1PU70_ConversionStep_cff import *

iterTracking = cms.Sequence(InitialStep*
                            HighPtTripletStep*
                            LowPtQuadStep*
                            LowPtTripletStep*
                            DetachedQuadStep*
                            MixedTripletStep*
                            PixelPairStep*
                            TobTecStep*
                            earlyGeneralTracks*
                            muonSeededStep*
                            preDuplicateMergingGeneralTracks*
                            generalTracksSequence*
                            ConvStep*
                            conversionStepTracks
                            )
