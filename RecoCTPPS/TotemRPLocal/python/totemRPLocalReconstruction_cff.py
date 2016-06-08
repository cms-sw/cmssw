import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import *
XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("Geometry/VeryForwardData/data/2016_ctpps_15sigma_margin0/RP_Dist_Beam_Cent.xml")

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
