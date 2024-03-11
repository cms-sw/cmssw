import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTDiphoton3023IsoCaloIdL1SeededInnerSequence = cms.Sequence(hltEgammaCandidatesL1Seeded+hltEgammaClusterShapeL1Seeded+hltEgammaHGCALIDVarsL1Seeded+hltEgammaHoverEL1Seeded+hltEgammaEcalPFClusterIsoL1Seeded+hltEgammaHGCalLayerClusterIsoL1Seeded+hltEgammaHcalPFClusterIsoL1Seeded)
