import FWCore.ParameterSet.Config as cms

# pixel track producer
from RecoHI.HiTracking.HIPixel3ProtoTracks_cfi import *

# vertex finding algorithms
from RecoHI.HiTracking.HIPixelMedianVertex_cfi import *
from RecoHI.HiTracking.HIPixelAdaptiveVertex_cfi import *

hiPixelVertices = cms.Sequence( hiPixel3ProtoTracks * (hiPixelMedianVertex + hiPixelAdaptiveVertex) )