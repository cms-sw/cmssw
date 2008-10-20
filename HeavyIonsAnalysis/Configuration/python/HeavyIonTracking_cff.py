import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from RecoPixelVertexing.PixelLowPtUtilities.common_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.firstStep_cff import *

pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.7
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1

pixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 1.5
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2

ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 6
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt           = 2.0

globalPrimTracks.Fitter = 'KFFittingSmoother'
globalPrimTracks.useHitsSplitting = True

firstStep = cms.Sequence(pixel3ProtoTracks
						* pixelVertices
						* pixel3PrimTracks
						* primSeeds
						* primTrackCandidates
						* globalPrimTracks )
heavyIonTracking = cms.Sequence(firstStep)

hiTrackingWithOfflineBeamSpot = cms.Sequence(offlineBeamSpot*trackerlocalreco*heavyIonTracking)


