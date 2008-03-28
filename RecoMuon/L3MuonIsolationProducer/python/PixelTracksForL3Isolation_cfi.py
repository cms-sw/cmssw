import FWCore.ParameterSet.Config as cms

import copy
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
pixelTracksForL3Isolation = copy.deepcopy(pixelTracks)
pixelTracksForL3Isolation.RegionFactoryPSet.ComponentName = 'IsolationRegionAroundL3Muon'
pixelTracksForL3Isolation.RegionFactoryPSet.RegionPSet = cms.PSet(
    deltaPhiRegion = cms.double(0.24),
    TrkSrc = cms.InputTag("L3Muons"),
    originHalfLength = cms.double(15.0),
    deltaEtaRegion = cms.double(0.24),
    vertexZDefault = cms.double(0.0),
    vertexSrc = cms.string(''),
    originRadius = cms.double(0.2),
    vertexZConstrained = cms.bool(False),
    ptMin = cms.double(1.0)
)

