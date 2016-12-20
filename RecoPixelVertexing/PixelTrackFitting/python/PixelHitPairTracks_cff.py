import FWCore.ParameterSet.Config as cms

from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelHitPairTracks_cfi import *
ttrhbwr.StripCPE = 'Fake'

