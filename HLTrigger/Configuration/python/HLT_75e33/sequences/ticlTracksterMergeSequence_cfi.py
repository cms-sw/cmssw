import FWCore.ParameterSet.Config as cms

from ..modules.ticlTrackstersMerge_cfi import *

ticlTracksterMergeSequence = cms.Sequence(ticlTrackstersMerge)
