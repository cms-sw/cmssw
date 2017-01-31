import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import *

# reco hit production
from RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi import ctppsDiamondRecHits

# local track fitting
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTrackFitter_cfi import ctppsDiamondLocalTrack

ctppsDiamondLocalReconstruction = cms.Sequence(
    ctppsDiamondRecHits
    * ctppsDiamondLocalTrack
)
