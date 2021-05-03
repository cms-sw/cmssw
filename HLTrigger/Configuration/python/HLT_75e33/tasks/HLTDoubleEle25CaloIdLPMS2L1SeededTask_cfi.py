import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTDoubleEle25CaloIdLPMS2L1SeededTask = cms.Task(hltEgammaCandidatesL1Seeded, hltEgammaClusterShapeL1Seeded, hltEgammaHGCALIDVarsL1Seeded, hltEgammaHoverEL1Seeded)
