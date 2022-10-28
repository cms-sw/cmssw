import FWCore.ParameterSet.Config as cms

from ..modules.hbhereco_cfi import *

hcalGlobalRecoTask = cms.Task(
    hbhereco
)
