import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTPhoton100OpenL1SeededTask = cms.Task(
    hltEgammaCandidatesL1Seeded,
    hltEgammaHGCALIDVarsL1Seeded,
    hltEgammaHoverEL1Seeded
)
