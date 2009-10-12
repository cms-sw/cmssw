import FWCore.ParameterSet.Config as cms

import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
hiProtoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()

hiProtoTracks.RegionFactoryPSet.ComponentName = cms.string('HITrackingRegionForPrimaryVtxProducer')
regPSet = hiProtoTracks.RegionFactoryPSet.RegionPSet

regPSet.directionXCoord = cms.double(1.0)
regPSet.directionYCoord = cms.double(1.0)
regPSet.directionZCoord = cms.double(0.0)
regPSet.ptMin = cms.double(0.5)
regPSet.originRadius = cms.double(0.1)
regPSet.siPixelRecHits = cms.string("siPixelRecHits")            




