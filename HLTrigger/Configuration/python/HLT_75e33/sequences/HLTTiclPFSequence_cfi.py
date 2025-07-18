import FWCore.ParameterSet.Config as cms

from ..modules.hltPfTICL_cfi import *

HLTTiclPFSequence = cms.Sequence(hltPfTICL)
