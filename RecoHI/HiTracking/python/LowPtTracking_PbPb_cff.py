import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HighPtTracking_PbPb_cff import *

hiPixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.9
hiPixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1
hiPixel3PrimTracks.FilterPSet.ptMin = 0.9
CkfBaseTrajectoryFilter_block.minPt = 0.9




