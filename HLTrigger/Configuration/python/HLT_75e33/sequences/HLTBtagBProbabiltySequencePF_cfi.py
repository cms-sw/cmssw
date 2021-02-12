import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepBLifetimeTagInfosPF_cfi import *
from ..modules.hltPfJetBProbabilityBJetTags_cfi import *

HLTBtagBProbabiltySequencePF = cms.Sequence(hltDeepBLifetimeTagInfosPF+hltPfJetBProbabilityBJetTags)
