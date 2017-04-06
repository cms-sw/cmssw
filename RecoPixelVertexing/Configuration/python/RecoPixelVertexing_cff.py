import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexing = cms.Sequence(PixelLayerTriplets*pixelTracksSequence*pixelVertices)

# For LowPU and Phase1PU70
PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
)
_recopixelvertexing_LowPU_Phase1PU70 = recopixelvertexing.copy()
_recopixelvertexing_LowPU_Phase1PU70.replace(PixelLayerTriplets, PixelLayerTripletsPreSplitting)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(recopixelvertexing, _recopixelvertexing_LowPU_Phase1PU70)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(recopixelvertexing, _recopixelvertexing_LowPU_Phase1PU70)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(recopixelvertexing, _recopixelvertexing_LowPU_Phase1PU70)
