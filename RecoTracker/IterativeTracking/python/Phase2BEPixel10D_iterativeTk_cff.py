import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.Phase1PU140_InitialStep_cff import *
from RecoTracker.IterativeTracking.Phase2PU140Pixel10D_HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase2PU140Pixel10D_LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase2PU140Pixel10D_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase2PU140Pixel10D_DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase2PU140Pixel10D_PixelPairStep_cff import *
from RecoTracker.FinalTrackSelectors.Phase1PU140_earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.Phase2BE_MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.Phase2BE_preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.Phase2BE_ConversionStep_cff import *

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
