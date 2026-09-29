import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersRecoveryL1Seeded_cfi import *
from ..modules.hltTiclTrackstersRecoveryL1Seeded_cfi import *

HLTTiclTrackstersRecoverySequenceL1Seeded = cms.Sequence(hltFilteredLayerClustersRecoveryL1Seeded+hltTiclTrackstersRecoveryL1Seeded)
