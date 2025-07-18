import FWCore.ParameterSet.Config as cms

from ..modules.hltTiclCandidate_cfi import *

HLTTiclCandidateSequence = cms.Sequence(hltTiclCandidate)
