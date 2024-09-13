import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersPassThrough_cfi import *
from ..modules.hltTiclTrackstersPassthrough_cfi import *

HLTTiclTrackstersPassthroughSequence = cms.Sequence(hltFilteredLayerClustersPassthrough+hltTiclTrackstersPassthrough)
