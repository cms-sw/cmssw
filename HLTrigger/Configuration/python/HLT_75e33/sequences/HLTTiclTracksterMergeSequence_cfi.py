import FWCore.ParameterSet.Config as cms

from ..modules.hltTiclTrackstersMerge_cfi import *

HLTTiclTracksterMergeSequence = cms.Sequence(hltTiclTrackstersMerge)
