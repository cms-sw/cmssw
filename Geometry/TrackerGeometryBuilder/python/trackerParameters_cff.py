import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.trackerParameters_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(trackerParameters,
                fromDD4Hep = cms.bool(True),
)
