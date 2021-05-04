import FWCore.ParameterSet.Config as cms

from Geometry.HcalTestBeamData.hcalTB02XtalParameters_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hcalTB02XtalParameters,
                fromDD4Hep = cms.bool(True),
)

hcalTB02HcalParameters = hcalTB02XtalParameters.clone(
    name  = cms.string("HcalHits"),
)
