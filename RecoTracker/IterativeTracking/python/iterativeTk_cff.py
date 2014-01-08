import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

from RecoTracker.IterativeTracking.PixelLessStep_cff import *
#from RecoTracker.IterativeTracking.PixelLessTripletStep_cff import *

from RecoTracker.IterativeTracking.TobTecStep_cff import *
#from RecoTracker.IterativeTracking.TobTecHybridStep_cff import *

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

iterTracking = cms.Sequence(InitialStep*
                            DetachedTripletStep*
                            LowPtTripletStep*
                            PixelPairStep*
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
