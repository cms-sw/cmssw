import FWCore.ParameterSet.Config as cms

from Geometry.HcalTestBeamData.hcalTB06BeamParameters_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hcalTB06BeamParameters,
                fromDD4Hep = cms.bool(True),
)
