import FWCore.ParameterSet.Config as cms

from ..modules.hltHbhereco_cfi import *

hcalGlobalRecoTask = cms.Task(
    hltHbhereco
)
