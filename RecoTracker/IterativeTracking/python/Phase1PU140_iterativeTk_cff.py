import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.Phase1PU140_InitialStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU140_HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU140_LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU140_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU140_DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU140_PixelPairStep_cff import *
from RecoTracker.FinalTrackSelectors.Phase1PU140_earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.Phase1PU70_MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.Phase1PU70_preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.Phase1PU140_ConversionStep_cff import *

iterTracking = cms.Sequence(InitialStep*
                            HighPtTripletStep*
                            LowPtQuadStep*
                            LowPtTripletStep*
                            DetachedQuadStep*
                            PixelPairStep*
                            earlyGeneralTracks*
                            muonSeededStep*
                            preDuplicateMergingGeneralTracks*
                            generalTracksSequence*
                            ConvStep*
                            conversionStepTracks
                            )
