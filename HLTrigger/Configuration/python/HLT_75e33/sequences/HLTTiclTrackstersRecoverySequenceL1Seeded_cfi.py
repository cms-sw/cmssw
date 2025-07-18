import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersRecoveryL1Seeded_cfi import *
from ..modules.hltTiclTrackstersRecoveryL1Seeded_cfi import *

HLTTiclTrackstersRecoverySequence = cms.Sequence(hltFilteredLayerClustersRecovery+hltTiclTrackstersRecovery)
