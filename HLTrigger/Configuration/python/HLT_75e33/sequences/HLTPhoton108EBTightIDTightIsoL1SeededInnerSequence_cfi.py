import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTPhoton108EBTightIDTightIsoL1SeededInnerSequence = cms.Sequence(hltEgammaCandidatesL1Seeded+hltEgammaClusterShapeL1Seeded+hltEgammaHoverEL1Seeded+hltEgammaEcalPFClusterIsoL1Seeded+hltEgammaHcalPFClusterIsoL1Seeded)
