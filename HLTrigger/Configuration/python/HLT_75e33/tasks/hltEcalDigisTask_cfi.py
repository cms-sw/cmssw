import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalDigis_cfi import *

hltEcalDigisTask = cms.Task(
    hltEcalDigis
)
