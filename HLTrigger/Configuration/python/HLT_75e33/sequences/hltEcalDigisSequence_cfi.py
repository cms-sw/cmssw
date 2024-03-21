import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalDigis_cfi import *

hltEcalDigisSequence = cms.Sequence(hltEcalDigis)
