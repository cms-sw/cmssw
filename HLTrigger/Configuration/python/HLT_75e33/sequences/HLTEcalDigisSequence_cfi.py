import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalDigis_cfi import *

HLTEcalDigisSequence = cms.Sequence(hltEcalDigis)
