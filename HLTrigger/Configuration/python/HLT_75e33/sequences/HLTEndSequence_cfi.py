import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolEnd_cfi import *

HLTEndSequence = cms.Sequence(hltBoolEnd)
