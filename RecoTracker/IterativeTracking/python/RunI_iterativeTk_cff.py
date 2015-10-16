import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.RunI_InitialStep_cff import *
from RecoTracker.IterativeTracking.RunI_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.RunI_PixelPairStep_cff import *
from RecoTracker.IterativeTracking.RunI_DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.RunI_MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.RunI_PixelLessStep_cff import *
from RecoTracker.IterativeTracking.RunI_TobTecStep_cff import *
from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

photonConvTrajSeedFromSingleLeg.OrderedHitsFactoryPSet.maxElement = 10000
photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.MaxNumberOfCosmicClusters = 150000
photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.MaxNumberOfPixelClusters = 20000
photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.cut = "strip < 150000 && pixel < 20000 && (strip < 20000 + 7* pixel)"

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

