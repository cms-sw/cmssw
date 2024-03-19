import FWCore.ParameterSet.Config as cms

from ..modules.siPhase2Clusters_cfi import *

HLTDoLocalStripSequence = cms.Sequence(siPhase2Clusters)
