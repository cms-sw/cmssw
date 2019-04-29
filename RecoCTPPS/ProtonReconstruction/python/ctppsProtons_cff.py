import FWCore.ParameterSet.Config as cms

from RecoCTPPS.ProtonReconstruction.ctppsProtons_cfi import *

# TODO: remove these lines once conditions data are available in DB
from CalibPPS.ESProducers.ctppsOpticalFunctions_cff import *
ctppsProtons.lhcInfoLabel = ctppsLHCInfoLabel
