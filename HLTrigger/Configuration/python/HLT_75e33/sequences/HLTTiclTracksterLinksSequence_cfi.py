import FWCore.ParameterSet.Config as cms

from ..modules.hltTiclTracksterLinks_cfi import *

HLTTiclTracksterLinksSequence = cms.Sequence(hltTiclTracksterLinks)
