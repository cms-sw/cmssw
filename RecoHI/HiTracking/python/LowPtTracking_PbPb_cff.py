import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HighPtTracking_PbPb_cff import *

hiTrackingRegionWithVertex.RegionPSet.ptMin = 0.9
hiTrackingRegionWithVertex.RegionPSet.originRadius = 0.1
hiFilter.ptMin = 0.9
CkfBaseTrajectoryFilter_block.minPt = 0.9




