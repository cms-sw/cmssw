import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersRecovery_cfi import *
from ..modules.hltTiclTrackstersRecovery_cfi import *

HLTTiclTrackstersRecoverySequence = cms.Sequence(hltFilteredLayerClustersRecovery+hltTiclTrackstersRecovery)
