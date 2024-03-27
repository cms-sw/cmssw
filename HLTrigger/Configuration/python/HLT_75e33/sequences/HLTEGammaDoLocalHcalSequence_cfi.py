import FWCore.ParameterSet.Config as cms

from ..modules.hltHbhereco_cfi import *
from ..modules.hltHcalDigis_cfi import *

HLTEGammaDoLocalHcalSequence = cms.Sequence(hltHcalDigis+hltHbhereco)
