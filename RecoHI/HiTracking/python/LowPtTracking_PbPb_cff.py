import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HighPtTracking_PbPb_cff import *

pixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.9
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1
ckfBaseTrajectoryFilter.filterPset.minPt = 0.9

lowPtHITracking = cms.Sequence(pixel3ProtoTracks
							  * pixel3Vertices
							  * pixel3PrimTracks
							  * primSeeds
							  * primTrackCandidates
							  * globalPrimTracks)




