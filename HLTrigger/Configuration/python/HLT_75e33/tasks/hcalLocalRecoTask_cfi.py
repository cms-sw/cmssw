import FWCore.ParameterSet.Config as cms

from ..modules.hfprereco_cfi import *
from ..modules.hfreco_cfi import *
from ..modules.horeco_cfi import *

hcalLocalRecoTask = cms.Task(
    hfprereco,
    hfreco,
    horeco
)
