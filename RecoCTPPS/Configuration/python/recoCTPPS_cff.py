import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff import *
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer
from RecoCTPPS.ProtonReconstruction.ctppsProtons_cfi import *

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *

# TODO: remove these lines once LHCInfo available in DB
from CalibPPS.ESProducers.ctppsLHCInfo_cff import *
ctppsProtons.lhcInfoLabel = ctppsLHCInfoESSource.label

# TODO: remove these lines once optical functions available in DB
from CalibPPS.ESProducers.ctppsOpticalFunctions_cff import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ctppsLHCInfoESSource.label

recoCTPPS = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer *
    ctppsProtons
)
