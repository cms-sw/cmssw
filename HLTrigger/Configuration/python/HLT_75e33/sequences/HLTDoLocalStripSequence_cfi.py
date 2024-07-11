import FWCore.ParameterSet.Config as cms

from ..modules.hltSiPhase2Clusters_cfi import *

HLTDoLocalStripSequence = cms.Sequence(hltSiPhase2Clusters)
