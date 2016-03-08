import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.InitialStepPreSplitting_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import *

# Phase1PU70 specific iterations
from RecoTracker.IterativeTracking.Phase1PU70_HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_LowPtQuadStep_cff import *
from RecoTracker.IterativeTracking.Phase1PU70_DetachedQuadStep_cff import *

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

iterTracking = cms.Sequence(InitialStepPreSplitting*
                            InitialStep*
                            DetachedTripletStep*
                            LowPtTripletStep*
                            PixelPairStep*
                            MixedTripletStep*
                            PixelLessStep*
                            TobTecStep*
			    JetCoreRegionalStep *	
                            earlyGeneralTracks*
                            muonSeededStep*
                            preDuplicateMergingGeneralTracks*
                            generalTracksSequence*
                            ConvStep*
                            conversionStepTracks
                            )

from Configuration.StandardSequences.Eras import eras
eras.trackingPhase1.toReplaceWith(iterTracking, cms.Sequence(
    InitialStep +
    HighPtTripletStep +
    LowPtQuadStep +
    LowPtTripletStep +
    DetachedQuadStep +
    MixedTripletStep +
    PixelPairStep +
    TobTecStep +
    earlyGeneralTracks +
    muonSeededStep +
    preDuplicateMergingGeneralTracks +
    generalTracksSequence +
    ConvStep +
    conversionStepTracks
))
