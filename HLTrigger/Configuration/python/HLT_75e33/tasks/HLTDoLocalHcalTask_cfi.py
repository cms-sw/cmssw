import FWCore.ParameterSet.Config as cms

from ..modules.hltHbhereco_cfi import *
from ..modules.hltHcalDigis_cfi import *
from ..modules.hltHfprereco_cfi import *
from ..modules.hltHfreco_cfi import *
from ..modules.hltHoreco_cfi import *

HLTDoLocalHcalTask = cms.Task(hltHbhereco, hltHcalDigis, hltHfprereco, hltHfreco, hltHoreco)
