import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.pixelTracks_cfi import pixelTracks as _pixelTracks
from RecoTauTag.HLTProducers.trackingRegionsFromBeamSpotAndL2Tau_cfi import trackingRegionsFromBeamSpotAndL2Tau

# Note from new seeding framework migration
# Previously the TrackingRegion was set as a parameter of PixelTrackProducer
# Now the TrackingRegion EDProducer must be inserted in a sequence, and set as an input to HitPairEDProducer

pixelTracksL2Tau = _pixelTracks.clone(
    passLabel = 'pixelTracksL2Tau'
)
