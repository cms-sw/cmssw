import FWCore.ParameterSet.Config as cms

from ..modules.hltHbhereco_cfi import *
from ..modules.hltHfprereco_cfi import *
from ..modules.hltHfreco_cfi import *
from ..modules.hltHoreco_cfi import *

HLTDoLocalHcalSequence = cms.Sequence(hltHbhereco+hltHoreco+hltHfprereco+hltHfreco)
