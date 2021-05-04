import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTPhoton100EBTightIDTightIsoOpenL1SeededTask = cms.Task(
    hltEgammaCandidatesL1Seeded,
    hltEgammaClusterShapeL1Seeded,
    hltEgammaEcalPFClusterIsoL1Seeded,
    hltEgammaHcalPFClusterIsoL1Seeded,
    hltEgammaHoverEL1Seeded
)
