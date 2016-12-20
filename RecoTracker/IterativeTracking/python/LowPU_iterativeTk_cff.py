import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.LowPU_InitialStep_cff import *
from RecoTracker.IterativeTracking.LowPU_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPU_PixelPairStep_cff import *
from RecoTracker.IterativeTracking.LowPU_DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPU_MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPU_PixelLessStep_cff import *
from RecoTracker.IterativeTracking.LowPU_TobTecStep_cff import *
from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

iterTracking = cms.Sequence(InitialStep*
                            LowPtTripletStep*
                            PixelPairStep*
                            DetachedTripletStep*
                            MixedTripletStep*
                            PixelLessStep*
                            TobTecStep*
                            earlyGeneralTracks*
                            muonSeededStep*
                            preDuplicateMergingGeneralTracks*
                            generalTracksSequence*
                            ConvStep*
                            conversionStepTracks
                            )
