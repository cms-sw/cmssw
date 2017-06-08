import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexing = cms.Sequence(PixelLayerTriplets*pixelTracksSequence*pixelVertices)

# For LowPU
PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
)
_recopixelvertexing_LowPU = recopixelvertexing.copy()
_recopixelvertexing_LowPU.replace(PixelLayerTriplets, PixelLayerTripletsPreSplitting)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(recopixelvertexing, _recopixelvertexing_LowPU)
