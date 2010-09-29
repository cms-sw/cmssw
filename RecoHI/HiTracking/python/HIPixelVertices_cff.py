import FWCore.ParameterSet.Config as cms

# pixel cluster vertex finder
from RecoHI.HiTracking.HIPixelClusterVertex_cfi import *

# pixel track producer
from RecoHI.HiTracking.HIPixel3ProtoTracks_cfi import *

# fast vertex finding 
from RecoHI.HiTracking.HIPixelMedianVertex_cfi import *

# selected pixel tracks
from RecoHI.HiTracking.HISelectedProtoTracks_cfi import *

# accurate vertex finding
from RecoHI.HiTracking.HIPixelAdaptiveVertex_cfi import *

# selection of best primary vertex
from RecoHI.HiTracking.HISelectedVertex_cfi import *

hiPixelVertices = cms.Sequence(hiPixelClusterVertex
                                * hiPixel3ProtoTracks 
                                * hiPixelMedianVertex 
                                * hiSelectedProtoTracks 
                                * hiPixelAdaptiveVertex
                                * bestHiVertex )
