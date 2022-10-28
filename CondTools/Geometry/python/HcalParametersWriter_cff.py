import FWCore.ParameterSet.Config as cms

from CondTools.Geometry.HcalParametersWriter_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(HcalParametersWriter,
                fromDD4hep = cms.bool(True)
)
