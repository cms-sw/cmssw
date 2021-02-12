import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepBLifetimeTagInfosPF_cfi import *
from ..modules.hltPfJetProbabilityBJetTags_cfi import *

HLTBtagProbabiltySequencePF = cms.Sequence(hltDeepBLifetimeTagInfosPF+hltPfJetProbabilityBJetTags)
