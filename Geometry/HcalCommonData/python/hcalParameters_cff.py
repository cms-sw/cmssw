import FWCore.ParameterSet.Config as cms

from Geometry.HcalCommonData.hcalParameters_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hcalParameters,
                fromDD4hep = cms.bool(True),
)
