import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import *
XMLIdealGeometryESSource.geomXMLFiles.append("Geometry/VeryForwardData/data/RP_Garage/RP_Dist_Beam_Cent.xml")

# clusterization
from RecoCTPPS.TotemRPLocal.totemRPClusterProducer_cfi import *

# reco hit production
from RecoCTPPS.TotemRPLocal.totemRPRecHitProducer_cfi import *

# non-parallel pattern recognition
from RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi import *

# local track fitting
from RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi import *

totemRPLocalReconstruction = cms.Sequence(
    totemRPClusterProducer *
    totemRPRecHitProducer *
    totemRPUVPatternFinder *
    totemRPLocalTrackFitter
)
