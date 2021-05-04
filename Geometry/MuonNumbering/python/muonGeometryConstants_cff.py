import FWCore.ParameterSet.Config as cms

from Geometry.MuonNumbering.muonGeometryConstants_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(muonGeometryConstants,
                fromDD4Hep = cms.bool(True),
)
