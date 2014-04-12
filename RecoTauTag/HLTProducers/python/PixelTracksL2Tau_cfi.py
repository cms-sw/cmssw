import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
from RecoTauTag.HLTProducers.TrackingRegionsFromBeamSpotAndL2Tau_cfi import *

pixelTracksL2Tau = pixelTracks.clone()
pixelTracksL2Tau.RegionFactoryPSet = cms.PSet(
    TrackingRegionsFromBeamSpotAndL2TauBlock,
    ComponentName = cms.string( "TrackingRegionsFromBeamSpotAndL2Tau" )
)
pixelTracksL2Tau.passLabel  = cms.string('pixelTracksL2Tau')

