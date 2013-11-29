import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *

#from RecoTracker.IterativeTracking.PixelLessTripletStep_cff import *
#from RecoTracker.IterativeTracking.TobTecHybridStep_cff import *

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

earlyGeneralTracks.selectedTrackQuals[0] = cms.InputTag("initialStep")
earlyGeneralTracks.selectedTrackQuals[5] = cms.InputTag("pixelLessStep")

photonConvTrajSeedFromSingleLeg.OrderedHitsFactoryPSet.maxElement = 40000
photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.MaxNumberOfCosmicClusters = 400000
photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.MaxNumberOfPixelClusters = 40000
photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.cut = "strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)"

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
