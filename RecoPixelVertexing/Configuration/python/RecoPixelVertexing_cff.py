import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexing = cms.Sequence(PixelLayerTriplets*pixelTracks*pixelVertices)

# For LowPU tracking
PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
)
_recopixelvertexing_trackingLowPU = recopixelvertexing.copy()
_recopixelvertexing_trackingLowPU.replace(PixelLayerTriplets, PixelLayerTripletsPreSplitting)
eras.trackingLowPU.toReplaceWith(recopixelvertexing, _recopixelvertexing_trackingLowPU)
