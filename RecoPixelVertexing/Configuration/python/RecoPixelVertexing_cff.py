import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexing = cms.Sequence(PixelLayerTriplets*pixelTracks*pixelVertices)

# For Phase1PU70
PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
)
_recopixelvertexing_Phase1PU70 = recopixelvertexing.copy()
_recopixelvertexing_Phase1PU70.replace(PixelLayerTriplets, PixelLayerTripletsPreSplitting)
eras.trackingPhase1PU70.toReplaceWith(recopixelvertexing, _recopixelvertexing_Phase1PU70)
